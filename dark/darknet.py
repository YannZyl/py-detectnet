# -*- coding: utf-8 -*-
import time
from utils.loader import WeightLoader
from .darkop import create_darkop
from utils.cfg_process import cfg_yielder

class DarkNet(object):
    
    def __init__(self, cfg_file, weight_file):
        self.src_cfg = cfg_file
        self.src_bin = weight_file
        # parsing yolo config
        src_parsed = self.parse_cfg(self.src_cfg)
        self.src_meta, self.src_layers = src_parsed
        # load yolo weights
        self.load_model()
        
    def parse_cfg(self, cfg_file):
        cfg_layers = cfg_yielder(cfg_file)
        meta, layers = dict(), list()
        for i, info in enumerate(cfg_layers):
            if i == 0:
                meta = info
                continue
            new = create_darkop(*info)
            layers.append(new)
        print('Parsing done. Total #{} layers'.format(len(layers)))
        return meta, layers
    
    def load_model(self):
        print('Loading weights: {}'.format(self.src_bin))
        start = time.time()
        wgts_loaders = WeightLoader(self.src_bin, self.src_layers)
        for layer in self.src_layers:
            layer.load(wgts_loaders)
        stop = time.time()
        print('Loading done. Finish in {}s'.format(stop-start))