# -*- coding: utf-8 -*-
from net.tfnet import TFNet

args = dict({
    'cfg_file': 'files/yolo_yann.cfg',
    'model_file': 'files/yolo_yann_89000.weights',
    'label_file': 'files/voc.names',
    'test': '/home/zyl8129/Desktop/images20_100',
    'output': 'output',
    'save': True,
    'gpu': 0.8,
    'train': False,
    'trainer': 'rmsprop',
    'img_dir':'/home/yann/Documents/python/HumanDetect_CNN/data/TRAIN/IMAGES_TRAIN',
    'ann_dir':'/home/yann/Documents/python/HumanDetect_CNN/data/TRAIN/ANNOTATIONS_TRAIN',
    'lr': 1e-5,
    'keep': 20,
    'batch': 16,
    'epoch': 100
})

net = TFNet(args)
net.predict()
