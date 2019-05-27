#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import chainer
import random
import pickle

from glob import glob
from os import path


class CelebA_Dataset(chainer.dataset.DatasetMixin):

    def __init__(self, cfg_mod, train):
        self.train = train
        self.cfg_mod = cfg_mod
        self.dataset = []

        print 'Converting CelebA Dataset...'
        if self.train:
            label = cfg_mod.train_data
        else:
            label = cfg_mod.test_data

        att_data = self.convert_attribute(cfg_mod.data_ann)

        f_list = open(cfg_mod.data_list, 'r')
        for idx, line in enumerate(f_list):
            ipath, raw_no = line[:-2].split(' ')
            sp_no = int(raw_no)
            for item_label in label:
                if item_label == sp_no:
                    img_full_path = path.join(cfg_mod.data_img_dir, ipath)
                    self.dataset.append([img_full_path, att_data[idx + 1]])

        print '  - # of samples :', len(self.dataset)
        print '\n'


    def convert_attribute(self, ann_path):
        f_ann = open(ann_path, 'r')
        data = []
        for idx, line in enumerate(f_ann):
            itemList = line[:-2].split(' ')
            if idx == 1:
                with open('attribute_name.pkl', mode='wb') as f:
                    pickle.dump(itemList, f)

            else:
                filter_item = [x for x in itemList if not x == '']
                item_data = []
                for item_filter in filter_item[1:]:
                    raw_label = int(item_filter)
                    if raw_label == -1:
                        raw_label = 0
                    item_data.append(raw_label)

                data.append(item_data)
        return data


    def get_path(self, i):
        data_path, att = self.dataset[i]
        return data_path


    def __len__(self):
        return len(self.dataset)


    def get_example(self, i):
        data_path, att = self.dataset[i]
        image = cv2.imread(data_path)
        r_img = cv2.resize(image, (self.cfg_mod.INPUT_COLS, self.cfg_mod.INPUT_ROWS))
        r_img = np.asarray(r_img, dtype=np.float32)
        r_img -= [103.939, 116.779, 123.68]
        trans_img = r_img.transpose((2, 0, 1))

        trans_img = np.asarray(trans_img, dtype=np.float32)
        att = np.asarray(att, dtype=np.int32).reshape((1, self.cfg_mod.CLASS))

        return trans_img, att

