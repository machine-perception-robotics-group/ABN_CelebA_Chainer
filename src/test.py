#coding: utf-8
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import six
import ctypes
import chainer
from chainer import Variable, optimizers, cuda, serializers, link
import time
import os
import imp
import random
import cPickle
import logging
import cv2

# from chainer.functions import caffe
from celeba import CelebA_Dataset
from os import path
from glob import glob
import cPickle as pickle


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


def get_model_optimizer(result_folder, cfg_mod):
    '''
    学習するネットワークモデルの構築
    パラメータの更新方法を選択
    '''

    model_fn = path.basename(cfg_mod.SRC_MODEL)

    # モデルのパラメータを取得
    useModel   = path.join(result_folder, cfg_mod.EVAL_PARAM_PATH)

    InNet_p = path.join(result_folder, cfg_mod.SRC_MODEL)
    src_model = imp.load_source(model_fn.split('.')[0], InNet_p).src_model
    serializers.load_npz(useModel, src_model)
    # serializers.load_npz(useModel, src_model)
    src_model.train = False
    # モデルをGPUに保存
    if cfg_mod.GPU_FLAG >= 0:
        src_model.to_gpu()

    return src_model


def normalize_confusion_matrix(raw_matrix, result_folder, class_num):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    RANGE = np.linspace(0.0, 100.0, 5, endpoint=True)
    norm_matrix = np.zeros((class_num, class_num))
    f_cm = open(path.join(result_folder, 'confusion_matrix.txt'), 'w')
    for class_index, class_item in enumerate(raw_matrix):
        sum_class = np.sum(class_item)
        norm_vec = class_item / float(sum_class)
        for item_vec in norm_vec:
            f_cm.write(str(item_vec) + ' ')
        f_cm.write('\n')
        norm_matrix[class_index] = norm_vec

    con = ax.imshow(norm_matrix, vmax = 1.0, vmin = 0.0, interpolation='nearest')
    cm = plt.colorbar(con, ticks = RANGE)
    plt.savefig(path.join(result_folder, 'figure.png'))


if __name__ == '__main__':
    args = parse_arguments()
    cfg_mod = load_module(args.src_train_config_path)
    result_folder = path.dirname(args.src_train_config_path)
    np.random.seed(cfg_mod.SEED)
    tp = 0
    tn = 0
    acc_tab = np.zeros((cfg_mod.CLASS, 2))

    with open("attribute_name.pkl", mode="rb") as f:
        class_name = pickle.load(f)
    class_name = class_name[:-1]

    # GPUの使用判定
    if cuda.available and cfg_mod.GPU_FLAG >= 0:
        cuda.get_device(cfg_mod.GPU_FLAG).use()

    test_data = CelebA_Dataset(cfg_mod=cfg_mod, train=False)
    sample_num = test_data.__len__()
    chainer.config.train = False
    raw_c_matrix = np.zeros((cfg_mod.CLASS, cfg_mod.CLASS))

    # 更新する方のネットワークモデルを取得
    src_model = get_model_optimizer(result_folder, cfg_mod)

    f_txt = open(path.join(result_folder, 'recog_result.txt'), 'w')
    xp = cuda.cupy if cfg_mod.GPU_FLAG >= 0 and cuda.available else np
    for index_sample in range(sample_num):
        print 'Loading data :', test_data.get_path(index_sample), index_sample
        v_img = cv2.imread(test_data.get_path(index_sample))
        v_img = cv2.resize(v_img, (cfg_mod.INPUT_COLS, cfg_mod.INPUT_ROWS))

        item_x, item_y = test_data.get_example(index_sample)
        item_x = np.array([item_x])

        x_batch = xp.asarray(item_x, dtype=np.float32)
        x_batch /= 255.
        y_batch = xp.asarray(item_y, dtype=np.int32)
        vx_batch = Variable(x_batch)
        vy_batch = Variable(y_batch)

        src_model(vx_batch, vy_batch)
        act = src_model.pred
        attention = src_model.attention

        out_dir = path.join('output', '{0:06d}'.format(index_sample))
        f_txt.write(out_dir + ',')
        if not path.exists(out_dir):
            os.mkdir(out_dir)
        for index_act, item_act in enumerate(act):
            np_act = chainer.cuda.to_cpu(item_act.data)
            int_act = int(np.argmax(np_act))
            if (int_act == 1) and (item_y[0, index_act] == 1):
                acc_tab[index_act, 0] += 1
                f_txt.write(class_name[index_act] + ',')
            elif (int_act == 0) and (item_y[0, index_act] == 0):
                acc_tab[index_act, 1] += 1
            item_attention = chainer.cuda.to_cpu(attention.data[0, index_act])
            # print np.min(item_attention), np.max(item_attention)

            resize_att = cv2.resize(item_attention, (cfg_mod.INPUT_COLS, cfg_mod.INPUT_ROWS)) * 255.
            cv2.imwrite('stock.png', resize_att)
            vis_map = cv2.imread('stock.png', 0)
            jet_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)
            jet_map = cv2.add(v_img, jet_map)

            out_path = path.join(out_dir, '{0:06d}'.format(index_sample) + '_' + class_name[index_act] + '.png')
            cv2.imwrite(out_path, jet_map)
        cv2.imwrite(path.join(out_dir, 'input.png'), v_img)
        f_txt.write('\n')

    att_acc = np.sum(acc_tab, axis=1) / float(sample_num)
    acc_f = open(path.join(result_folder, 'accuracy.txt'), 'w')
    for index_acc, item_acc in enumerate(att_acc):
        acc_f.write(str(item_acc * 100) + '\n')
        print class_name[index_acc], ' : ', item_acc * 100, '%'

    # normalize_confusion_matrix(raw_c_matrix, result_folder, cfg_mod.CLASS)
    print 'Average accuracy :', np.mean(att_acc) * 100, '%'




    logging.info('OVER')














