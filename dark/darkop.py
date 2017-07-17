# -*- coding: utf-8 -*-
import numpy as np

class Layer(object):
    def __init__(self, *args):
        self._signature = list(args)
        self.type = list(args)[0]
        self.number = list(args)[1]

        self.w = dict() # weights
        self.h = dict() # placeholders
        self.wshape = dict() # weight shape
        self.wsize = dict() # weight size
        self.setup(*args[2:]) # set attr up
        for var in self.wshape:
            shp = self.wshape[var]
            size = np.prod(shp)
            self.wsize[var] = size

    def load(self, src_loader):
        wdict = self.load_weights(src_loader)
        if wdict is not None:
            self.recollect(wdict)

    def load_weights(self, src_loader):
        val = src_loader([self])
        if val is None: return None
        else: return val.w

    @property
    def signature(self):
        return self._signature

    def recollect(self, w): self.w = w
    def setup(self, *args): pass
    def finalize(self): pass
        
class maxpool_layer(Layer):
    def setup(self, ksize, stride, pad):
        self.stride = stride
        self.ksize = ksize
        self.pad = pad

class route_layer(Layer):
    def setup(self, routes):
        self.routes = routes

class reorg_layer(Layer):
    def setup(self, stride):
        self.stride = stride
        
class convolutional_layer(Layer):
    def setup(self, ksize, c, n, stride, 
              pad, batch_norm, activation):
        self.batch_norm = bool(batch_norm)
        self.activation = activation
        self.stride = stride
        self.ksize = ksize
        self.pad = pad
        self.train_phrase = False
        self.dnshape = [n, c, ksize, ksize] # darknet shape
        self.wshape = dict({
            'biases': [n], 
            'kernel': [ksize, ksize, c, n]
        })
        if self.batch_norm:
            self.wshape.update({
                'moving_variance'  : [n], 
                'moving_mean': [n], 
                'gamma' : [n]
            })
            self.h['is_training'] = {
                'feed': True,
                'dfault': False,
                'shape': ()
            }
    def finalize(self, _):
        """deal with darknet"""
        kernel = self.w['kernel']
        if kernel is None: return
        kernel = kernel.reshape(self.dnshape)
        kernel = kernel.transpose([2,3,1,0])
        self.w['kernel'] = kernel
        
darkops = {
    'maxpool': maxpool_layer,
    'convolutional': convolutional_layer,
    'route': route_layer,
    'reorg': reorg_layer,
}

def create_darkop(ltype, num, *args):
    op_class = darkops.get(ltype, Layer)
    return op_class(ltype, num, *args)