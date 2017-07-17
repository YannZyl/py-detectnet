# -*- coding: utf-8 -*-
import tensorflow as tf
from .baseop import Baseop
from tensorflow.contrib import slim

class Route(Baseop):
    def forward(self):
        routes = self.lay.routes
        out = list()
        for r in routes:
            # search the layer
            this = self.inp
            while this.lay.number != r:
                this = this.inp
                assert this is not None, 'can not route the layer:{}'.format(r)
            out += [this.out]
        self.out = tf.concat(out, 3)

class Maxpool(Baseop):
    def forward(self):
        self.out = tf.nn.max_pool(self.inp.out, padding='SAME',
                                  ksize=[1]+[self.lay.ksize]*2+[1],\
                                  strides=[1]+[self.lay.stride]*2+[1],\
                                  name = self.scope)

class Leaky(Baseop):
    def forward(self):
        inp = self.inp.out
        self.out = tf.where(inp>0, inp, 0.1*inp, name=self.scope)
        
class Convolutional(Baseop):
    def forward(self):
        pad = [[self.lay.pad, self.lay.pad]]*2
        temp = tf.pad(self.inp.out,[[0,0]]+pad+[[0,0]])
        temp = tf.nn.conv2d(temp, self.lay.w['kernel'], padding = 'VALID', 
            name = self.scope, strides = [1] + [self.lay.stride] * 2 + [1])
        if self.lay.batch_norm:
            temp = self.batchnorm(temp)
        self.out = tf.nn.bias_add(temp, self.lay.w['biases'])
        #self.out = temp
    
    def batchnorm(self, inp):
        args = dict({
            'center': False, 'scale': True,
            'epsilon': 1e-5, 'scope':self.scope,
            'updates_collections' : None,
            'is_training': self.lay.h['is_training'],
            'param_initializers': self.lay.w
        })
        return slim.batch_norm(inp, **args)

class Reorg(Baseop):
    # method 1 
    def _forward(self):
        inp = self.inp.out
        shape = inp.get_shape().as_list()
        _, h, w, c = shape
        s = self.lay.stride
        out = list()
        for i in range(int(h/s)):
            row_i = list()
            for j in range(int(w/s)):
                si, sj = s*i, s*j
                boxij = inp[:,si:si+s,sj:sj+s,:]
                flatij = tf.reshape(boxij, [-1,1,1,c*s*s])
                row_i += [flatij]
            out += [tf.concat(row_i, 2)]
        self.out = tf.concat(out, 1)
    # method 2: tensorflow api
    def forward(self):
        inp = self.inp.out
        s = self.lay.stride
        self.out = tf.extract_image_patches(inp,ksizes=[1,s,s,1],strides=[1,s,s,1],
                                        rates=[1,1,1,1], padding='VALID')