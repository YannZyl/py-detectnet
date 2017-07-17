# -*- coding: utf-8 -*-
import os
import dark
import numpy as np

class WeightLoader(object):
    VAR_LAYER = ['convolutional']
    _W_ORDER = dict({ # order of param flattened into .weights file
        'convolutional': [
            'biases','gamma','moving_mean','moving_variance','kernel'
        ]
    })
    def __init__(self, *args):
        self.src_key = []
        self.vals = []
        self.load(*args)
    
    def __call__(self, key):
        for idx in range(len(key)):
            val = self.find(key, idx)
            if val is not None: return val
            return None
        
    def find(self, key, idx):
        up_to = min(len(self.src_key), 4)
        for i in range(up_to):      
            key_b = self.src_key[i]
            if key_b[idx:] == key[idx:]:
                return self.yields(i)
        return None

    def yields(self, idx):
        del self.src_key[idx]
        temp = self.vals[idx]
        del self.vals[idx]
        return temp
   
    def load(self, path, src_layers):
        self.src_layers = src_layers
        walker = WeightsWarker(path)
        for i, layer in enumerate(src_layers):
            if layer.type not in self.VAR_LAYER: continue
            self.src_key.append([layer])
            
            if walker.eof: new = None       
            else:
                args = layer.signature
                new = dark.darknet.create_darkop(*args)
            self.vals.append(new)
            
            if new is None: continue
            order = self._W_ORDER[new.type]
            for par in order:
                if par not in new.wshape: continue
                val = walker.walk(new.wsize[par])
                new.w[par] = val
            new.finalize(walker.transpose)        
        """   
        if walker.path is not None:
            assert walker.offset == walker.size, \
                    'except {} bytes, found {}'.format(walker.offset, walker.size)
            print 'Successfully identified {} bytes'.format(walker.offset)
        """
# walker, read data from model file
class WeightsWarker(object):
    def __init__(self, path):
        self.eof = False
        self.path = path
        if path is None:
            self.eof = True
            return
        else:
            self.size = os.path.getsize(path)
            major, minor, revision, seen = np.memmap(path, dtype = '({})i4,'.format(4), 
                                                     offset=0, mode='r', shape=())
            self.transpose = major > 1000 or minor > 1000
            self.offset = 16
            
    def walk(self, size):
        if self.eof:
            return None
        end_point = self.offset + 4*size
        assert end_point <= self.size, 'over-read {}'.format(self.path)
        float32_1D_array = np.memmap(self.path, shape=(), mode='r', offset=self.offset,
                                     dtype='({})float32,'.format(size))
        self.offset = end_point
        if end_point == self.size:
            self.eof = True
        return float32_1D_array