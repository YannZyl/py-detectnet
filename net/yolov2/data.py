# -*- coding: utf-8 -*-
import os
import numpy as np
from copy import deepcopy
from utils.xml_process import extract_from_xmlfolder

def parse(self, exclusive = False):
    data = extract_from_xmlfolder(self.args['ann_dir'])
    return data

def _batch(self, chunk):
    meta = self.meta
    S, _, _ = meta['out_size']
    B = meta['num']
    C, labels = meta['classes'], meta['labels']

    # preprocess
    jpg = chunk['filename']
    inp_h, inp_w, inp_c = self.meta['inp_size']
    h, w, c = chunk['size']
    allobj_ = chunk['allobjs']
    allobj = deepcopy(allobj_)
    path = os.path.join(self.args['img_dir'], jpg)
    img = self.preprocess(path, allobj)
    # Calculate regression target
    cellx = 1. * inp_w / S
    celly = 1. * inp_h / S
    # label
    probs = np.zeros([S*S,C])
    confs = np.zeros([S*S,B])
    coord = np.zeros([S*S,B,4])
    #proid = np.zeros([S*S,C])
    prear = np.zeros([S*S,4])
    for obj in allobj:
        bbox = obj['bndbox']
        label = obj['name']
        # adjust bounding box
        scalex = 1. * inp_w / w
        scaley = 1. * inp_h / h
        bbox[0] = bbox[0] * scalex
        bbox[1] = bbox[1] * scaley
        bbox[2] = bbox[2] * scalex
        bbox[3] = bbox[3] * scaley
        # 
        centerx = .5*(bbox[0]+bbox[2]) #xmin, xmax
        centery = .5*(bbox[1]+bbox[3]) #ymin, ymax
        
        cx = centerx / cellx
        cy = centery / celly
        if cx >= S or cy >= S: 
            return None, None
        bbox[2] = float(bbox[2]-bbox[0]) / inp_w
        bbox[3] = float(bbox[3]-bbox[1]) / inp_h
        bbox[2] = np.sqrt(bbox[2])
        bbox[3] = np.sqrt(bbox[3])
        bbox[0] = cx - np.floor(cx) # centerx
        bbox[1] = cy - np.floor(cy) # centery
        bbox += [int(np.floor(cy) * S + np.floor(cx))]
        #print bbox
        
        # fill into label
        probs[bbox[4], :] = [0.] * C
        probs[bbox[4], labels.index(label)] = 1.
        #proid[bbox[4], :] = [1] * C
        coord[bbox[4], :, :] = [bbox[0:4]] * B
        prear[bbox[4],0] = bbox[0] - bbox[2]**2 * .5 * S # xleft
        prear[bbox[4],1] = bbox[1] - bbox[3]**2 * .5 * S # yup
        prear[bbox[4],2] = bbox[0] + bbox[2]**2 * .5 * S # xright
        prear[bbox[4],3] = bbox[1] + bbox[3]**2 * .5 * S # ybot
        confs[bbox[4], :] = [1.] * B

    # Finalise the placeholders' values
    upleft   = np.expand_dims(prear[:,0:2], 1)
    botright = np.expand_dims(prear[:,2:4], 1)
    probs = np.expand_dims(probs, 1)
    wh = botright - upleft; 
    area = wh[:,:,0] * wh[:,:,1]
    probs    = np.concatenate([probs] * B, 1)
    #print probs.shape
    upleft   = np.concatenate([upleft] * B, 1)
    botright = np.concatenate([botright] * B, 1)
    areas = np.concatenate([area] * B, 1)

    # value for placeholder at input layer
    inp_feed_val = img
    # value for placeholder at loss layer 
    loss_feed_val = {
        'probs': probs, 'confs': confs, 
        'coord': coord,# 'proid': proid,
        'areas': areas, 'upleft': upleft, 
        'botright': botright
    }

    return inp_feed_val, loss_feed_val
    
def shuffle(self):
    batch = self.args['batch']
    data = self.parse()
    size = len(data)
    
    print('Dataset of {} instance(s)'.format(size))
    if batch > size:
        self.args['batch'] = batch = size
    batch_per_epoch = int(size/batch)
    
    for i in range(self.args['epoch']):
        shuffle_idx = np.random.permutation(np.arange(size))
        print('epoch-->',i)
        for b in range(batch_per_epoch):
            x_batch = list()
            feed_batch = dict()
            
            for j in range(b*batch, b*batch+batch):
                train_instance = data[shuffle_idx[j]]
                
                inp, new_feed = self._batch(train_instance)
                
                if inp is None: continue
                x_batch += [np.expand_dims(inp, 0)]
                for key in new_feed:
                    new = new_feed[key]
                    old = feed_batch.get(key, np.zeros((0,)+new.shape))
                    feed_batch[key] = np.concatenate([old, [new]])
            x_batch = np.concatenate(x_batch, 0)
            yield x_batch, feed_batch
        print('Finish {} epoch(es)'.format(i+1))