#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random

from os import path
from glob import glob


# GPUのフラグ
GPU_FLAG = 1
EPOCH = 10
BATCH_SIZE = 8
E_BATCH_SIZE = 1      # 基本的に'1'
# パラメータを保存するタイミング
SNAPSHOT = 4
# 乱数のシード設定
SEED = 1701
CLASS = 40

# サンプルの設定
CHANNEL = 3
INPUT_COLS = 224
INPUT_ROWS = 224
ATT_TH = 0.0
# 学習サンプル
data_root = path.join('..', 'data')
data_list = path.join(data_root, 'list_eval_partition.txt')
data_ann  = path.join(data_root, 'Anno', 'list_attr_celeba.txt')
data_img_dir = path.join(data_root, 'img_align_celeba')
train_data = [0, 1]
test_data  = [2]

# パラメータの更新の設定
OPT_PARAM = 'MomentumSGD'
TRAIN_RATE = 0.01
LR_STEP = 4
LR_DROP = 0.1
WEIGHT_DECAY = 0.0001
MOMENTUM = 0.9
BETA1 = 0.9
BETA2 = 0.9999
EPS = 1e-08

FREQUENCY = 2
EVAL_PARAM_PATH = 'model_snapshot_10.npz'

# パラメータを更新するモデルの指定
SRC_MODEL = 'Ftuning_model.py'

