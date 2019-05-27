#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cupy as cp
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers
from chainer.links.model.vision.resnet import ResNet101Layers


class BottleNeckA(chainer.Chain):

    def __init__(self, in_size, ch, out_size, stride=2):
        super(BottleNeckA, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_size, ch, 1, stride, 0, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(
                ch, ch, 3, 1, 1, initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(ch)
            self.conv3 = L.Convolution2D(
                ch, out_size, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn3 = L.BatchNormalization(out_size)

            self.conv4 = L.Convolution2D(
                in_size, out_size, 1, stride, 0,
                initialW=initialW, nobias=True)
            self.bn4 = L.BatchNormalization(out_size)

    def __call__(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))
        h2 = self.bn4(self.conv4(x))

        return F.relu(h1 + h2)


class BottleNeckB(chainer.Chain):

    def __init__(self, in_size, ch):
        super(BottleNeckB, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_size, ch, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(
                ch, ch, 3, 1, 1, initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(ch)
            self.conv3 = L.Convolution2D(
                ch, in_size, 1, 1, 0, initialW=initialW, nobias=True)
            self.bn3 = L.BatchNormalization(in_size)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))

        return F.relu(h + x)


class Block(chainer.Chain):

    def __init__(self, layer, in_size, ch, out_size, stride=2):
        super(Block, self).__init__()
        self.add_link('a', BottleNeckA(in_size, ch, out_size, stride))
        for i in range(1, layer):
            self.add_link('b{}'.format(i), BottleNeckB(out_size, ch))
        self.layer = layer

    def __call__(self, x):
        h = self.a(x)
        for i in range(1, self.layer):
            h = self['b{}'.format(i)](h)

        return h


class Multitask_Block(chainer.Chain):

    def __init__(self, att_num):
        self.att_num = att_num
        super(Multitask_Block, self).__init__()
        with self.init_scope():
            self.res5 = Block(3, 1024, 512, 2048)
            for i in range(self.att_num):
                self.add_link('att{}'.format(i), L.Linear(2048, 2))
        pretrained_model = ResNet101Layers()
        self.res5.copyparams(pretrained_model.res5)
        del(pretrained_model)


    def __call__(self, x, attentions):
        am, an, ay, ax = attentions.shape
        ret_pred = []
        for i in range(self.att_num):
            item_attention = attentions[:, i].reshape(am, 1, ay, ax)
            sh = F.scale(x, item_attention, axis=0)
            sh = self.res5(sh)
            sh = F.average_pooling_2d(sh, 7, stride=1)
            sh = self['att{}'.format(i)](sh)
            ret_pred.append(sh)

        return ret_pred


class Ftuning_model(chainer.Chain):

    insize = 224

    def __init__(self):
        super(Ftuning_model, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                3, 64, 7, 2, 3, initialW=initializers.HeNormal(), nobias=True)
            self.bn1 = L.BatchNormalization(64)
            self.res2 = Block(3, 64, 64, 256, 1)
            self.res3 = Block(4, 256, 128, 512)
            self.res4 = Block(23, 512, 256, 1024)
            # self.res5 = Block(3, 1024, 512, 2048)
            self.multitask = Multitask_Block(40)
            self.fc = L.Linear(2048, 80)

            self.att_res5 = Block(3, 1024, 512, 2048, 1)
            self.att_out=L.Convolution2D(2048, 40, 1, stride=1, initialW=initializers.HeNormal(), nobias=True)


    def __call__(self, x, t_att):
        h = self.bn1(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)

        gh = self.att_res5(h)
        gh = self.att_out(gh)
        sm, sn, sy, sx = gh.shape
        gap = F.reshape(F.max_pooling_2d(gh, (sy, sx)), (sm, 1, 40))
        self.attention = gh

        resps = self.multitask(h, gh)

        if chainer.config.train:
            self.g_loss = F.sigmoid_cross_entropy(gap, t_att)
            t_att = t_att.transpose(2, 1, 0)
            self.r_loss = 0

            for index_resp, item_resp in enumerate(resps):
                self.r_loss += F.softmax_cross_entropy(item_resp, t_att[index_resp, 0])
            self.r_loss /= 40

            self.loss = self.r_loss + self.g_loss
            # self.accuracy = F.accuracy(h, t_att)
            chainer.report({
                'g_loss': self.g_loss,
                'r_loss': self.r_loss,
                'loss': self.loss,
                # 'accuracy': self.accuracy,
            }, self)
            return self.loss
        else:
            self.pred = resps
            return resps


src_model = Ftuning_model()

