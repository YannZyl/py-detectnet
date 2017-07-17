# -*- coding: utf-8 -*-
import time
from . import flow
import tensorflow as tf
from dark.darknet import DarkNet
from .yolov2.yolov2 import YOLOv2
from .ops.layerop import create_layerop
from .ops.baseop import Identity


class TFNet(object):  
    _TRAINER = dict({
		'rmsprop': tf.train.RMSPropOptimizer,
		'adadelta': tf.train.AdadeltaOptimizer,
		'adagrad': tf.train.AdagradOptimizer,
		'adagradDA': tf.train.AdagradDAOptimizer,
		'momentum': tf.train.MomentumOptimizer,
		'adam': tf.train.AdamOptimizer,
		'ftrl': tf.train.FtrlOptimizer,
	})
    predict = flow.predict
    train = flow.train
    
    def __init__(self, args):
        self.args = args
        self.cfg_file = args['cfg_file']
        self.model_file = args['model_file']
        # construct darknet
        darknet = DarkNet(self.cfg_file, self.model_file)
        self.darknet = darknet
        self.ntrain = len(darknet.src_layers)
        self.num_layer = len(darknet.src_layers)
        self.meta = darknet.src_meta
        self.yolov2 = YOLOv2(self.meta, args)
        # build net
        start = time.time()
        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            self.build_forward()
            self.setup_meta_ops()
        print('Finished in {}s\n'.format(time.time() - start))
    
    def build_forward(self):
        # placeholder
        inp_size = [None] + self.meta['inp_size']
        self.inp = tf.placeholder(tf.float32, inp_size, 'input')
        self.feed = dict() # other placeholders
        # build the forward
        flow = Identity(self.inp)
        for i, layer in enumerate(self.darknet.src_layers):
            args = [layer, flow, i, self.feed]
            flow = create_layerop(*args)
        #self.top = flow
        self.out = tf.identity(flow.out, name='output')
        
    def build_train_op(self):
        self.yolov2.build_loss(self.out)
        print('Building {} train op'.format(self.meta['model']))
        optimizer = self._TRAINER[self.args['trainer']](self.args['lr'])
        gradients = optimizer.compute_gradients(self.yolov2.loss)
        self.train_op = optimizer.apply_gradients(gradients)
    
    def setup_meta_ops(self):
        cfg = dict({
			'allow_soft_placement': False,
			'log_device_placement': False})
        utility = min(self.args['gpu'], 1.)
        if utility > 0.0:
            cfg['gpu_options'] = tf.GPUOptions(
                per_process_gpu_memory_fraction = utility)
            cfg['allow_soft_placement'] = True
        else: 
            cfg['device_count'] = {'GPU': 0}

        if self.args['train']: 
            self.build_train_op()
        self.sess = tf.Session(config = tf.ConfigProto(**cfg))
        self.sess.run(tf.global_variables_initializer())
        
        if not self.ntrain: return
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep = self.args['keep'])
            