#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
How to run：
Multi Layer Perceptron
>> python train.py --model MLP.py
Deep Convolutional Neural Network
>> python train.py --model Cifar10.py
GPU mode
>> python train.py --gpu 0
'''

import argparse
import imp
import numpy as np
import os
import six
import cv2
import random
import sys, time
import pickle
import matplotlib
matplotlib.use('Agg')  # isort:skip
import sklearn
import chainer
import chainer.links as L
from celeba import CelebA_Dataset

from glob import glob
from os import path
from scipy import linalg
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable
from chainer import training
from chainer.training import extensions
from chainer.links.model.vision.resnet import ResNet101Layers


def get_model_optimizer(result_folder, cfg_mod):
    '''
    学習するネットワークモデルの構築
    パラメータの更新方法を選択
    '''
    model_fn = path.basename(cfg_mod.SRC_MODEL)
    src_model = imp.load_source(model_fn.split('.')[0], path.join(result_folder, cfg_mod.SRC_MODEL)).src_model

    # モデルをGPUに保存
    #if cfg_mod.GPU_FLAG >= 0:
    #    src_model.to_gpu()

    # パラメータの更新方法を指定
    if cfg_mod.OPT_PARAM == 'AdaGrad':
        optimizer = optimizers.AdaGrad(lr=cfg_mod.TRAIN_RATE, eps=cfg_mod.EPS)
    elif cfg_mod.OPT_PARAM == 'MomentumSGD':
        optimizer = optimizers.MomentumSGD(lr=cfg_mod.TRAIN_RATE, momentum=cfg_mod.MOMENTUM)
    elif cfg_mod.OPT_PARAM == 'AdaDelta':
        optimizer = optimizers.AdaDelta(rho=cfg_mod.TRAIN_RATE, eps=cfg_mod.EPS)
    elif cfg_mod.OPT_PARAM == 'ADAM':
        optimizer = optimizers.Adam(alpha=cfg_mod.TRAIN_RATE, beta1=cfg_mod.BETA1, beta2=cfg_mod.BETA2, eps=cfg_mod.EPS)
    else:
        raise Exception('No optimizer is selected')
    optimizer.setup(src_model)

    if cfg_mod.WEIGHT_DECAY:
        optimizer.add_hook(chainer.optimizer.WeightDecay(cfg_mod.WEIGHT_DECAY))


    return src_model, optimizer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'src_train_config_path',
        help='source configuration file path'
    )

    return parser.parse_args()


def load_module(module_path):
    '''
    module_path: モジュールへのパス(フォルダ・拡張子含む)

    return ロードされたモジュール
    '''
    head, tail = path.split(module_path)
    module_name = path.splitext(tail)[0]
    info = imp.find_module(module_name, [head])
    return imp.load_module(module_name, *info)


def transform(data):
    img, label = data
    img = img / 255.

    return img, label


if __name__ == '__main__':
    args = parse_arguments()
    cfg_mod = load_module(args.src_train_config_path)
    result_folder = path.dirname(args.src_train_config_path)

    # ネットワークとoptimizerの構築
    model, optimizer = get_model_optimizer(result_folder, cfg_mod)

    pretrained_model = ResNet101Layers()
    model.conv1.copyparams(pretrained_model.conv1)
    model.res2.copyparams(pretrained_model.res2)
    model.res3.copyparams(pretrained_model.res3)
    model.res4.copyparams(pretrained_model.res4)
    model.res5.copyparams(pretrained_model.res5)


    # 学習＆評価サンプルの取得
    print('Convert training sample...')
    train_data = CelebA_Dataset(cfg_mod=cfg_mod, train=True)
    train_datas = chainer.datasets.TransformDataset(train_data, transform)
    test_data  = CelebA_Dataset(cfg_mod=cfg_mod, train=False)
    test_datas = chainer.datasets.TransformDataset(test_data, transform)

    # 学習と評価のループ
    train_iter = chainer.iterators.SerialIterator(train_datas, cfg_mod.BATCH_SIZE)
    test_iter = chainer.iterators.SerialIterator(test_datas, cfg_mod.BATCH_SIZE, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=cfg_mod.GPU_FLAG)
    trainer = training.Trainer(updater, (cfg_mod.EPOCH, 'epoch'), out=result_folder)

    trainer.extend(extensions.Evaluator(test_iter, model, device=cfg_mod.GPU_FLAG))
    trainer.extend(extensions.dump_graph('main/loss'))

    frequency = cfg_mod.EPOCH if cfg_mod.FREQUENCY == -1 else max(1, cfg_mod.FREQUENCY)
    trainer.extend(extensions.snapshot_object(model, 'model_snapshot_{.updater.epoch}.npz'), trigger=(frequency, 'epoch'))

    trainer.extend(extensions.LogReport(trigger=(20, 'iteration')))
    trainer.extend(
        extensions.PlotReport(['main/loss', 'main/r_loss', 'main/g_loss', 'main/accuracy'],
                                  'epoch', file_name='loss.png'))
    trainer.extend(extensions.ExponentialShift('lr', cfg_mod.LR_DROP),
                   trigger=(cfg_mod.LR_STEP, 'epoch'))
    trainer.extend(chainer.training.extensions.observe_lr(),
                   trigger=(20, 'iteration'))

    trainer.extend(
        extensions.PlotReport(
            ['main/accuracy', 'main/accuracy'],
            'epoch', file_name='accuracy.png'))

    trainer.extend(extensions.PrintReport(
        ['iteration', 'epoch', 'lr', 'main/loss', 'main/r_loss', 'main/g_loss', 'main/accuracy', 'elapsed_time']), trigger=(20,'iteration'))

    trainer.extend(extensions.ProgressBar(update_interval=10))

    # Resume from a snapshot
    # chainer.serializers.load_npz('snapshot_model.npz', trainer)

    trainer.run()

    print('OVER')




