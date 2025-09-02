# -*- coding: utf-8 -*-
"""
Created on Wed May 28 11:38:15 2025

@author: zhouyc
"""

import lightning as L
import torch
from torch.optim import lr_scheduler

from core.segmentor import build_segmentor
from .metrics import intersect_and_union, pre_eval_to_metrics
from .losses import CrossEntropyLoss, DiceLoss, Accuracy

def denormalize(tensor, 
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                inplace=True):
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean).to(tensor.device)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std).to(tensor.device)
        
    if tensor.dim() == 4:
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
    elif tensor.dim() == 3:
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
    else:
        raise ValueError
        
    if not inplace:
        tensor = tensor.clone()
        
    tensor.mul_(std).add_(mean)
    
    return torch.clamp(tensor, 0, 1)

class SegmentModule(L.LightningModule):
    def __init__(self, args, **kwargs):
        super(SegmentModule, self).__init__()
        self.args = args
        
        kwp = {'mlla_enable': not args.mlla_disable,
               'fft_enable': not args.fft_disable,
               'gate_enable': not args.gate_disable}
        self.model = build_segmentor(args, **kwp)
        
        self.save_hyperparameters() # write hyperparams into a file
        
        return
    
    # OVERLOAD: the forward pass of the lightning model
    def forward(self, image):
        outputs = self.model(image)
        return outputs
    
    # OVERLOAD: configure the optimizer 
    def configure_optimizers(self):
        param_dicts = [{"params": [p for _, p in self.named_parameters() if p.requires_grad]},]
        
        if self.args.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                param_dicts, #self.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
                momentum=self.args.momentum)
        elif self.args.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                param_dicts, #self.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay)
        elif self.args.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(
                param_dicts, #self.parameters(), 
                lr=self.args.lr,
                weight_decay=self.args.weight_decay)
        else:
            raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')
            
        if self.args.lr_sched.lower() == 'multistep':
            scheduler = lr_scheduler.MultiStepLR(
                optimizer, 
                milestones=self.args.milestones, 
                gamma=self.args.gamma
                )
        elif self.args.lr_sched.lower() == 'step' or self.args.lr_sched.lower() == 'steplr':
            scheduler = lr_scheduler.StepLR(
                optimizer, 
                step_size=self.args.step_size,
                gamma=self.args.gamma
                )
        elif self.args.lr_sched.lower() == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, 
                self.args.T_max
                )
        elif self.args.lr_sched.lower() == 'linear':
            scheduler = lr_scheduler.LinearLR(
                optimizer, 
                start_factor=self.args.start_factor, 
                end_factor=self.args.end_factor,
                total_iters=self.args.total_iters
                )
        elif self.args.lr_sched.lower() == 'onecycle':
            scheduler = lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.args.lr,
                total_steps=self.args.num_steps+100,
                pct_start=self.args.pct_start,
                cycle_momentum=self.args.cycle_momentum,
                anneal_strategy=self.args.anneal_strategy,
                )
        return [optimizer], [{'scheduler':scheduler, 'interval': self.args.lr_update_interval}]
    
    def metric_function(self, predicts, labels, work=None):
        outputs = list()
        for pred, label in zip(predicts, labels):
            pred = torch.argmax(pred, dim=0).cpu().numpy()
            label = label.cpu().numpy()
            output = intersect_and_union(pred, label,
                self.model.decoder.num_classes, self.args.ignore_index)
            outputs.append(output)
            
        return outputs
    
    def loss_function(self, predicts, labels):
        # loss
        CEL = CrossEntropyLoss(loss_weight=0.6)
        DL = DiceLoss(loss_weight=0.4, ignore_index=self.args.ignore_index)
        ACC = Accuracy(topk=1, ignore_index=self.args.ignore_index)
        
        loss = CEL(predicts, labels, ignore_index=self.args.ignore_index) \
            + DL(predicts, labels)
        acc = ACC(predicts, labels)
        
        return loss, acc
    
    # OVERLOAD: This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        image, label = batch['image'], batch['label']
        label = label.squeeze(1)
        mask = self(image)
        
        loss, acc = self.loss_function(mask, label)
        
        self.log('loss', loss.item(), on_step=True, prog_bar=True, logger=True)
        self.log('acc', acc.item(),  on_step=True, prog_bar=True, logger=True)
        
        return {'loss': loss}
    
    # OVERLOAD: 
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        image, label = batch['image'], batch['label']
        label = label.squeeze(1)
        mask = self(image)
        ious = self.metric_function(mask, label, 'val')
        
        self.val_outputs.extend(ious)
        
        return
    
    def on_validation_epoch_start(self):
        self.val_outputs = []
        return
    
    def log_metrics(self, metrics, work=None):
        for key in metrics:
            metric = metrics[key]
            if metric.ndim == 0:
                metric = [metric.item()]
            for i, v in enumerate(metric):
                self.log(('' if work is None else work) + key + ('' if len(metric)==1 else str(i)), 
                         v, prog_bar=True, logger=True)
        if 'IoU' in metrics:
            metric = metrics['IoU'].mean()
            self.log(('' if work is None else work) +'mIoU', metric, prog_bar=True, logger=True)
        return
    
    def on_validation_epoch_end(self):
        metrics = pre_eval_to_metrics(self.val_outputs, self.args.metrics)
        self.log_metrics(metrics, 'eval/')
        self.val_outputs = []
        return
    
    # OVERLOAD: 
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        image, label = batch['image'], batch['label']
        label = label.squeeze(1)
        mask = self(image)
        ious = self.metric_function(mask, label, 'test')
        self.test_outputs.extend(ious)
        
        return
    
    def on_test_epoch_start(self):
        self.test_outputs = []
        return
    
    def on_test_epoch_end(self):
        metrics = pre_eval_to_metrics(self.test_outputs, self.args.metrics)
        self.log_metrics(metrics, 'test/')
        self.test_outputs = []
        return
    
    # OVERLOAD: 
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        patch = batch['patch']
        pad = batch['pad']
        file = batch['file']
        full_size = batch['full_size']
        pos = batch['pos']
        
        mask = self(patch)
        mask = torch.argmax(mask, dim=1)
        
        patch = denormalize(patch)
        patch = patch * 255.0
        patch = patch.clip(0, 255).to(torch.uint8)
        
        output = {'patch': patch.cpu().numpy(),
                  'mask': mask.cpu().numpy(),
                  'pad': pad.cpu().numpy(),
                  'file': file,
                  'full_size': full_size.cpu().numpy(),
                  'pos': pos.cpu().numpy()
                  }
        
        return output