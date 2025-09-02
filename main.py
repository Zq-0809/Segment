# -*- coding: utf-8 -*-
"""
Created on Wed May 28 13:56:16 2025

@author: zhouyc
"""
import os
import cv2
import argparse
import torch
import lightning as L
import numpy as np
from datamodule.datamodule import SegmentDataModule
from module.segment_module import SegmentModule
from config import get_config


def arguments():
    parser = argparse.ArgumentParser(description='Disparity modal')
    
    parser.add_argument('--work', type=str, default='fit', 
                        choices=['fit', 'eval', 'test', 'predict'])
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    
    # MODAL
    parser.add_argument('--resume', 
                        default=r'.\logs\lightning_logs\version_31\checkpoints\last.ckpt', 
                        type=str,
                        help='resume from pretrained model or resume from unexpectedly terminated training')
    parser.add_argument('--pretrained-root', type=str, 
                        default='./checkpoints')
    parser.add_argument('--image-size', type=int, nargs='+', default=[512,512])
    
    parser.add_argument('--mlla-disable', action='store_true', default=False)
    parser.add_argument('--fft-disable', action='store_true', default=False)
    parser.add_argument('--gate-disable', action='store_true', default=False)
    parser.add_argument('--cfg', type=str, required=False, default='./cfgs/mff_s.yaml',
                        metavar="FILE", help='path to config file',)
    
    # OPTIMIZER
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0, type=float)
    parser.add_argument('--weight-decay', default=1e-2, type=float)
    parser.add_argument('--clip_max_norm', default=1.0, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--precision', default=None, type=str, help='float number precision')
    parser.add_argument('--lr-sched', type=str, default='onecycle')
    parser.add_argument('--lr-update-interval', type=str, default='step', 
                        choices=['epoch', 'step'], help='the unit of the scheduler\'s step size')
    parser.add_argument('--milestones', type=int, nargs='+', default=[20,40], # MultiStepLR
                        help='decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.') 
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='multiplicative factor of learning rate decay.') # MultiStepLR, StepLR
    parser.add_argument('--step-size', default=10, type=int,                  # StepLR
                        help='decays the learning rate of each parameter group by gamma every step_size epochs.')
    parser.add_argument('--T_max', default=100000, type=int,                 # CosineAnnealingLR
                        help='maximum number of iterations.')
    parser.add_argument('--start-factor', default=0.5, type=float,            # LinearLR
                        help='the number we multiply learning rate in the first epoch.') 
    parser.add_argument('--end-factor', default=1.0, type=float,              # LinearLR
                        help='the number we multiply learning rate at the end of linear changing process.')
    parser.add_argument('--total-iters', default=100, type=int,               # LinearLR
                        help='maximum number of iterations.')
    '''
    The 1cycle learning rate policy changes the learning rate after every batch. 
    step should be called after a batch has been used for training.
    '''
    # parser.add_argument('--total-steps', default=100000, type=int,          # OneCycleLR
    #                     help='the total number of steps in the cycle.')
    parser.add_argument('--pct-start', default=0.01, type=float,               # OneCycleLR
                        help='the percentage of the cycle (in number of steps) spent increasing the learning rate.')
    parser.add_argument('--anneal-strategy', type=str, default='linear',       # OneCycleLR
                        choices=['cos', 'linear'], help='specifies the annealing strategy.')
    parser.add_argument('--cycle-momentum', action='store_true',               # OneCycleLR
                        help='if True, momentum is cycled inversely to learning rate between ‘base_momentum’ and ‘max_momentum’.')
    
    # DATA
    parser.add_argument('--dataset', type=str, default='AWSD')
    parser.add_argument('--val-set', type=str, default='AWSD')
    parser.add_argument('--test-set', type=str, default='AWSD')
    parser.add_argument('--predict-set', type=str, default='AWSD')
    parser.add_argument('--data-root', type=str, default='../../RSD')
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--seed', default=42, type=int)
    
    # TRAINNING
    parser.add_argument('--monitor', default='eval/mIoU', type=str)
    parser.add_argument('--checkpoint_dir', default='./logs/', type=str, 
                        help='where to save the training log and models')
    parser.add_argument('--num_steps', default=160000, type=int,
                        help='stop training after this number of steps for grad-backward')    
    parser.add_argument('--grad_freq', default=1, type=int,
                        help='accumulates gradients over grad_freq batches before stepping the optimizer')
    parser.add_argument('--summary_freq', default=100, type=int, help='grad-backward steps')
    parser.add_argument('--val_freq', default=10000, type=int, help='grad-backward steps')
    parser.add_argument('--save_ckpt_freq', default=10000, type=int, 
                        help='number of training (grad-backward) steps between checkpoints.')
    
    # EVALUATING
    parser.add_argument('--metrics', type=str, nargs='+', 
                        default=['mIoU', 'mDice', 'mFscore'])
    parser.add_argument('--ignore-index', type=int, default=-100,
                        help='The label index to be ignored. If used, should be set to 255')
    parser.add_argument('--reduce-zero-label', action='store_true', default=False, 
                        help='Whether reduce all label value by 1. Usually used for datasets where 0 is background label. If used bg will be 255')
    
    # PREDICTING
    parser.add_argument('--output_dir', default='./output/', type=str, 
                        help='where to save the predictions')
    
    args = parser.parse_args()
    
    return args

def get_datamodule(args):
    datamodule = SegmentDataModule( 
                     dataset = args.dataset,
                     val_set = args.val_set,
                     test_set = args.test_set,
                     predict_set = args.predict_set,
                     image_width = args.image_size[1],
                     image_height = args.image_size[0],
                     batch_size = args.batch_size,
                     num_workers = args.num_workers,
                     data_root = args.data_root)
    
    return datamodule

def get_model(args):
    model = SegmentModule(args)
    return model

def save_predictions(predicts, args):
    output_path = args.output_dir
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    images = dict()
    labels = dict()
    for item in predicts:
        patches = item['patch']
        masks = item['mask']
        pads = item['pad']
        files = item['file']
        full_sizes = item['full_size']
        poses = item['pos']
        
        for patch, mask, pad, file, full_size, pos in zip(
                patches, masks, pads, files, full_sizes, poses):
            if pad[0] != 0 and pad[1] != 0:
                patch = patch[:, 0:-pad[0], 0:-pad[1]]
                mask = mask[0:-pad[0], 0:-pad[1]]
            elif pad[0] == 0 and pad[1] == 0:
                patch = patch
                mask = mask
            elif pad[0] == 0:
                patch = patch[:, :, :-pad[1]]
                mask = mask[:, :-pad[1]]
            else:
                patch = patch[:, :-pad[0], :]
                mask = mask[:-pad[0], :]
                
            if file not in images:
                images[file] = np.zeros((patch.shape[0], full_size[0], full_size[1]), dtype=patch.dtype)
            image = images[file]
            image[:, pos[0]:pos[2], pos[1]:pos[3]] = patch
            
            if file not in labels:
                labels[file] = np.zeros((full_size[0], full_size[1]), dtype=mask.dtype)
            label = labels[file]
            label[pos[0]:pos[2], pos[1]:pos[3]] = mask
        
    palette = [[0, 0, 0],   #background (RED, GREEN, BLUE)
               [255, 255, 0], #
               [0, 176, 0], #
               ]
    palette = np.array(palette, dtype=np.uint8)
    alpha = 0.3
    
    for key in images:
        image = images[key]
        image = image.transpose(1,2,0)
        label = labels[key]
        
        fn, _ = os.path.splitext(key)
        
        path = os.path.join(output_path, fn+'.jpg')
        cv2.imwrite(path, image[:,:,::-1]) # RGB==>BGR
        
        path = os.path.join(output_path, fn+'-mask.png')
        cv2.imwrite(path, label.astype(np.uint8))
        
        color_mask = palette[label]
        path = os.path.join(output_path, fn+'-maskviz.png')
        cv2.imwrite(path, color_mask[:,:,::-1]) # RGB==>BGR
        
        overlayed = (image * (1 - alpha) + color_mask * alpha).astype(np.uint8)
        path = os.path.join(output_path, fn+'-overlay.jpg')
        cv2.imwrite(path, overlayed[:,:,::-1]) # RGB==>BGR
        
        print('output', key)
        
    return

def main(args):
    datamodule = get_datamodule(args)
    model = get_model(args)
    
    # model params saving using Pytorch Lightning
    checkpoint_cb = L.pytorch.callbacks.ModelCheckpoint(
        monitor=args.monitor,
        filename=f'model_{args.dataset}' + '_{step:06d}',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=10,
        save_last=True,
        mode='max',
        save_on_train_epoch_end=False,
        every_n_train_steps=args.save_ckpt_freq,
        every_n_epochs=None
    )
    
    # trainer handles all loop details 
    trainer = L.Trainer(
        default_root_dir=args.checkpoint_dir,
        accelerator='gpu',
        precision=args.precision, #'16-mixed', #None
        devices='auto',
        strategy='auto',
        fast_dev_run=False, # for debugging purposes
        accumulate_grad_batches=args.grad_freq,
        gradient_clip_val=args.clip_max_norm,
        gradient_clip_algorithm='norm',
        num_sanity_val_steps=100, # runs a validation step before stating training
        max_steps=args.num_steps,
        val_check_interval=args.val_freq*args.grad_freq,
        check_val_every_n_epoch=None,        # must set to None
        callbacks=[checkpoint_cb],
        reload_dataloaders_every_n_epochs=0, # reload for sample order shuffle
        log_every_n_steps=args.summary_freq
    )
    
    # start training or ...
    if args.work == 'fit':
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        trainer.fit(model=model, datamodule=datamodule) #, ckpt_path=args.resume)
        print('The best model path:', checkpoint_cb.best_model_path)
        print('The best model IoU:', checkpoint_cb.best_model_score)
    elif args.work == 'eval':
        trainer.validate(model=model, datamodule=datamodule, 
                         ckpt_path=args.resume, verbose=True)
    elif args.work == 'test':
        trainer.test(model=model, datamodule=datamodule, 
                     ckpt_path=args.resume, verbose=True)
    elif args.work == 'predict':
        outputs = trainer.predict(model=model, datamodule=datamodule,
                        ckpt_path=args.resume, return_predictions =True)
        save_predictions(outputs, args)
        
    return

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')
    args = arguments()
    L.seed_everything(args.seed, workers=True)
    cfg = get_config(args)
    main(cfg)
    