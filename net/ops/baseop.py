# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

class Baseop(object):
    _SLIM = ['gamma', 'moving_mean', 'moving_variance']
    
    def __init__(self, layer, inp, num, feed):
        self.inp = inp # BaseOp
        self.num = num # int
        self.out = None # tf.Tensor
        self.lay = layer
        self.act = 'Load'
        self.scope = '{}-{}'.format(str(self.num), self.lay.type)
        # filled data into tensor
        self.convert(feed)
        # build operations
        self.forward()
        
    
    def forward(self):
        pass
    
    def convert(self, feed):
        for var in self.lay.wshape:
            self.wrap_variable(var)
        for ph in self.lay.h:
            self.wrap_pholder(ph, feed)

    def wrap_pholder(self, ph, feed):
        """wrap layer.h into placeholders"""
        phtype = type(self.lay.h[ph])
        if phtype is not dict: return
        sig = '{}/{}'.format(self.scope, ph)
        val = self.lay.h[ph] 

        self.lay.h[ph] = tf.placeholder_with_default(
            val['dfault'], val['shape'], name = sig)
        feed[self.lay.h[ph]] = val['feed']
        
    def wrap_variable(self, var):
        val = self.lay.w.get(var, None)
        # initial new params
        if var is None:
            shape = self.lay.wshape[var]
            args = [0., 1e-2, shape]
            if 'moving_mean' in var:
                val = np.zeros(shape)
            elif 'moving_variance' in var:
                val = np.ones(shape)
            else:
                val = np.random.normal(*args)
            self.lay.w[var] = var.astype(np.float32)
            self.act = 'Init'
        # load from wieghts file
        self.lay.w[var] = tf.constant_initializer(val)
        if var in self._SLIM: return
        with tf.variable_scope(self.scope):
            self.lay.w[var] = tf.get_variable(var,
            shape = self.lay.wshape[var],
            dtype = tf.float32,
            initializer = self.lay.w[var])    

class Identity(Baseop):
    def __init__(self, inp):
        self.inp = None
        self.out = inp