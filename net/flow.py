# -*- coding: utf-8 -*-
import os 
import time
import numpy as np
from sklearn.externals import joblib

_EXT = ['jpg', 'JPG', 'jpeg', 'JPEG']

def train(self):
    pass

def predict(self):
    test_dir = self.args['test']
    all_inp_ = os.listdir(test_dir)
    all_inp = list()
    for name in all_inp_:
        if name.split('.')[-1] in _EXT:
            all_inp.append(name)
    if all_inp == list():
        print('Failed to find ant test image im test directory')
    batch = min(self.args['batch'], len(all_inp))
    for j in range(len(all_inp)//batch):
        inp_feed = list()
        batch_inp = all_inp[j*batch:j*batch+batch]
        for inp in batch_inp:
            this_inp = os.path.join(test_dir, inp)
            this_inp = self.yolov2.preprocess(this_inp)
            expanded = np.expand_dims(this_inp, 0)
            inp_feed.append(expanded)
        
        feed_dict = {self.inp: np.concatenate(inp_feed,0)}
        print('Forwarding {} inputs...'.format(len(inp_feed)))
        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        
        #joblib.dump(out[0], '/home/yann/Desktop/1111/1.pkl')
        stop = time.time()
        print('Total time = {}s / {} inps = {} ips'.format(
            stop-start, len(inp_feed), len(inp_feed) / (stop-start+0.0)))
        
        print('Post processing {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        for i, prediction in enumerate(out):
            self.yolov2.postprocess(prediction, os.path.join(test_dir, all_inp[i]))
        stop = time.time(); 
        print('Total time = {}s / {} inps = {} ips'.format(
             stop-start, len(inp_feed), len(inp_feed) / (stop-start+0.0)))