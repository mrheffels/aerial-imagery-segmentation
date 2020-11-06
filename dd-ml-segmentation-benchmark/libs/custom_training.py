"""
    train.py - Sample implementation of a Dynamic Unet using FastAI
    2019 - Nicholas Pilkington, DroneDeploy
"""
import os
import sys
sys.path.append(os.getcwd())
from fastai.vision import *
from fastai.callbacks.hooks import *
from libs import inference
from libs import scoring
from libs.util import MySaveModelCallback, ExportCallback, MyCSVLogger, Precision, Recall, FBeta
from libs import datasets_fastai
from torch import nn
import torch.nn.functional as F

import wandb
from wandb.fastai import WandbCallback

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, reduction='elementwise_mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction is None:
            return F_loss
        else:
            return torch.mean(F_loss)

def train_model(dataset):
    """ Trains a DynamicUnet on the dataset """

    epochs = 40
    lr     = 1e-4
    size   = 300
    wd     = 1e-2
    bs     = 2 # reduce this if you are running out of GPU memory
    pretrained = True

    config = {
        'epochs' : epochs,
        'lr' : lr,
        'size' : size,
        'wd' : wd,
        'bs' : bs,
        'pretrained' : pretrained,
    }

    wandb.config.update(config)

    metrics = [
        Precision(average='weighted', clas_idx=1),
        Recall(average='weighted', clas_idx=1),
        FBeta(average='weighted', beta=1, clas_idx=1),
    ]

    data = datasets_fastai.load_dataset(dataset, size, bs)
    encoder_model = models.resnet50
    learn = unet_learner(data, encoder_model, path='models', metrics=metrics, wd=wd, bottle=True, pretrained=pretrained)
    learn.loss_fn = FocalLoss()
    callbacks = [
        WandbCallback(learn, log=None, input_type="images"),
        MyCSVLogger(learn, filename='baseline_model'),
        ExportCallback(learn, "baseline_model", monitor='f_beta'),
        MySaveModelCallback(learn, every='epoch', monitor='f_beta')
    ]

    learn.unfreeze()
    learn.fit_one_cycle(epochs, lr, callbacks=callbacks)
