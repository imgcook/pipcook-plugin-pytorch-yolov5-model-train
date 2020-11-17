import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))

import yaml
import logging
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda import amp
from tools.datasets import create_dataloader, preprocess
from tqdm import tqdm
import math
import numpy as np
import time
import evaluate

from tools.general import (set_logging,
    init_seeds, 
    check_dataset, 
    check_img_size, 
    torch_distributed_zero_first, 
    plot_labels, 
    labels_to_class_weights, 
    compute_loss,
    plot_images,
    fitness,
    check_anchors
)
from tools.torch_utils import select_device, ModelEMA

logger = logging.getLogger(__name__)

class obj(object):
  def __init__(self, d):
    for a, b in d.items():
      if isinstance(b, (list, tuple)):
          setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
      else:
          setattr(self, a, obj(b) if isinstance(b, dict) else b)

def train(modelWrapper, data, hyp, opt, device):
  model = modelWrapper.model
  ckpt = modelWrapper.config['ckpt']
  logger.info(f'Hyperparameters {hyp}')
  log_dir = opt.modelPath
  wdir = log_dir + '/weights'
  os.makedirs(wdir, exist_ok=True)
  last = wdir + '/last.pt'
  best = wdir + '/best.pt'
  results_file = log_dir + '/results.txt'

  epochs, batch_size, total_batch_size, weights, rank = \
    opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

  with open(log_dir + '/hyp-train.yaml', 'w') as f:
    yaml.dump(hyp, f, sort_keys=False)
  with open(log_dir + '/opt-train.yaml', 'w') as f:
    yaml.dump(vars(opt), f, sort_keys=False)

  # Configure
  cuda = device.type != 'cpu'
  init_seeds(2 + rank)

  with open(opt.data) as f:
    data_dict = yaml.load(f, Loader=yaml.FullLoader)
  with torch_distributed_zero_first(rank):
    check_dataset(data_dict)
  train_path = data_dict['train']
  test_path = data_dict['val']
  nc, names = (int(data_dict['nc']), data_dict['names'])
  assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)

  # Optimizer
  nbs = 64 
  accumulate = max(round(nbs / total_batch_size), 1)
  hyp['weight_decay'] *= total_batch_size * accumulate / nbs

  pg0, pg1, pg2 = [], [], []
  for k, v in model.named_parameters():
      v.requires_grad = True
      if '.bias' in k:
          pg2.append(v)
      elif '.weight' in k and '.bn' not in k:
          pg1.append(v)
      else:
          pg0.append(v)

  optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

  optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})
  optimizer.add_param_group({'params': pg2})
  logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
  del pg0, pg1, pg2

  lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp['lrf']) + hyp['lrf']  # cosine
  scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

  start_epoch, best_fitness = 0, 0.0
  # Optimizer
  if ckpt['optimizer'] is not None:
      optimizer.load_state_dict(ckpt['optimizer'])
      best_fitness = ckpt['best_fitness']

  # Results
  if ckpt.get('training_results') is not None:
      with open(results_file, 'w') as file:
          file.write(ckpt['training_results'])

  # Epochs
  start_epoch = ckpt['epoch'] + 1
  if epochs < start_epoch:
      logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                  (weights, ckpt['epoch'], epochs))
      epochs += ckpt['epoch']

  del ckpt

  # Image sizes
  gs = int(max(model.stride))
  imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]

  # DP mode
  if cuda and torch.cuda.device_count() > 1:
      model = torch.nn.DataParallel(model)

  # Exponential moving average
  ema = ModelEMA(model)

  dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                          hyp=hyp, augment=True)
  mlc = np.concatenate(dataset.labels, 0)[:, 0].max()
  nb = len(dataloader)
  assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

  ema.updates = start_epoch * nb // accumulate

  labels = np.concatenate(dataset.labels, 0)
  c = torch.tensor(labels[:, 0])
  plot_labels(labels, save_dir=log_dir)
  check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

  # Model parameters
  hyp['cls'] *= nc / 80.
  model.nc = nc
  model.hyp = hyp
  model.gr = 1.0
  model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)
  model.names = names

  # Start training
  t0 = time.time()
  nw = max(round(hyp['warmup_epochs'] * nb), 1e3)
  maps = np.zeros(nc)  # mAP per class
  results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
  scheduler.last_epoch = start_epoch - 1  # do not move
  scaler = amp.GradScaler(enabled=cuda)
  logger.info('Image sizes %g train, %g test\n'
              'Using %g dataloader workers\nLogging results to %s\n'
              'Starting training for %g epochs...' % (imgsz, imgsz_test, dataloader.num_workers, log_dir, epochs))
  logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'targets', 'img_size'))
  for epoch in range(start_epoch, epochs):
    logger.info('Epoch: ' + str(epoch))
    model.train()

    mloss = torch.zeros(4, device=device)  # mean losses
    pbar = enumerate(dataloader)
    
    optimizer.zero_grad()
    for i, (imgs, targets, paths, _) in pbar:
        ni = i + nb * epoch  # number integrated batches (since train start)
        imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

        # Warmup
        if ni <= nw:
            xi = [0, nw]  # x interp
            # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
            accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

        # Forward
        with amp.autocast(enabled=cuda):
            pred = model(imgs)  # forward
            loss, loss_items = compute_loss(pred, targets.to(device), model)  # loss scaled by batch_size
            if rank != -1:
                loss *= opt.world_size  # gradient averaged between devices in DDP mode

        # Backward
        scaler.scale(loss).backward()

        # Optimize
        if ni % accumulate == 0:
            scaler.step(optimizer)  # optimizer.step
            scaler.update()
            optimizer.zero_grad()
            if ema:
                ema.update(model)

        # Print
        mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
        mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
        s = ('%10s' * 2 + '%10.4g' * 6) % (
            '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])

        # Plot
        if ni < 3:
            f = str(('log_dir/train_batch%g.jpg' % ni))  # filename
            result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)


        # end batch ------------------------------------------------------------------------------------------------
    logger.info(s)
    # Scheduler
    lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
    scheduler.step()

    # DDP process 0 or single-GPU
    # mAP
    if ema:
        ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride'])
    final_epoch = epoch + 1 == epochs
    results, maps, times = evaluate.test(opt.data,
        batch_size=total_batch_size,
        imgsz=imgsz_test,
        model=ema.ema,
        single_cls=opt.single_cls,
        dataloader=dataloader,
        save_dir=log_dir,
        plots=epoch == 0 or final_epoch)
    # Write
    with open(results_file, 'a') as f:
        f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)

    # Update best mAP
    fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
    if fi > best_fitness:
        best_fitness = fi
    logger.info('Current Best Map: ' + str(fi))

    # Save model
    with open(results_file, 'r') as f:  # create checkpoint
        ckpt = {'epoch': epoch,
                'best_fitness': best_fitness,
                'training_results': f.read(),
                'model': ema.ema,
                'optimizer': None if final_epoch else optimizer.state_dict()}

    # Save last, best and delete
    torch.save(ckpt, last)
    if best_fitness == fi:
        torch.save(ckpt, best)
    del ckpt
    # end epoch ----------------------------------------------------------------------------------------------------
  return imgsz
# end training


  

def main(data, model, args):
  opt = obj({})
  opt.total_batch_size = 16 if not hasattr(args, 'batch_size') else args.batchSize
  opt.epochs = 300 if not hasattr(args, 'epochs') else args.epochs
  opt.batch_size = opt.total_batch_size
  opt.world_size = 1
  opt.global_rank = -1
  opt.hyp = os.path.join(os.path.dirname(__file__), 'config/hyp.scratch.yaml') 
  opt.device = ''
  opt.weights = 'yolov5s.pt'
  opt.single_cls = False
  opt.modelPath = args.modelPath
  opt.img_size = model.config['img_size']

  set_logging(opt.global_rank)

  opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))
  device = select_device(opt.device, batch_size=opt.batch_size)
  logger.info(opt)
  with open(opt.hyp) as f:
    hyp = yaml.load(f, Loader=yaml.FullLoader)
  dataconfig = preprocess(data)
  model.cfg = obj({})
  model.cfg.data = opt.data = dataconfig

  imgsz = train(model, data, hyp, opt, device)
  model.cfg.imgsz = imgsz
  sys.path.pop()
  return model
