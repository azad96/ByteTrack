#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "datasets/2022-10-06T16-34-42"
        self.images = "images"
        
        # self.train_ann = "coco_annotations.json"
        # self.val_ann = "coco_annotations.json"

        # self.train_ann = "coco_annotations_only_car.json"
        # self.val_ann = "coco_annotations_only_car.json"

        # self.train_ann = "subsampled_coco_annotations.json"
        # self.val_ann = "subsampled_coco_annotations.json"

        self.train_ann = "coco_annotations_only_car_no_empty_train.json"
        self.val_ann = "coco_annotations_only_car_no_empty_val.json"

        self.input_size = (640, 640)
        self.num_classes = 1

        self.max_epoch = 300
        self.data_num_workers = 4
        self.eval_interval = 1
        self.print_interval = 50