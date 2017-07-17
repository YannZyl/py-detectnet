# -*- coding: utf-8 -*-
import os
import cv2
import math
import numpy as np
from utils.box import BoundBox, box_iou
from utils.im_transform import imcv2_affine_trans, imcv2_recolor

def logistic(x):
    return 1. / (1. + np.exp(-x))

def softmax(x)   :
    e_x = np.exp(x-np.max(x))
    out = e_x / e_x.sum()
    return out 
    
def _fix(obj, dims, scale, offs):
    for i in range(1,5):
        dim = dims[(i+1)%2]
        off = offs[(i+1)%2]
        obj[i] = int(obj[i] * scale - off)
        obj[i] = max(min(obj[i], dim), 0)

def resize_input(self, im):
    h, w, c = self.meta['inp_size']
    imsz = cv2.resize(im, (w, h))
    imsz = imsz / 255.
    imsz = imsz[:,:,::-1]
    return imsz

def findboxes(self, net_out):
    meta = self.meta
    H, W, _ = meta['out_size']
    threshold = meta['thresh']
    C, B = meta['classes'], meta['num']
    anchors = meta['anchors']
    net_out = net_out.reshape([H, W, B, -1])
    
    boxes = list()
    for row in range(H):
        for col in range(W):
            for b in range(B):
                bx = BoundBox(C)
                bx.x,bx.y,bx.w,bx.h,bx.c = net_out[row, col, b, :5]
                bx.c = logistic(bx.c)
                bx.x = (col + logistic(bx.x)) / W
                bx.y = (row + logistic(bx.y)) / H
                bx.w = math.exp(bx.w) * anchors[2*b+0] / W
                bx.h = math.exp(bx.h) * anchors[2*b+1] / H
                classes = net_out[row, col, b, 5:]
                bx.probs = softmax(classes) * bx.c
                bx.probs *= bx.probs > threshold
                boxes.append(bx)
    # nms
    for c in range(C):
        # sorted according to probability of class c
        boxes = sorted(boxes, key=lambda box: box.probs[c], reverse=True)
        for i in range(len(boxes)):
            boxi = boxes[i]
            if boxi.probs[c] == 0: 
                continue
            # remove redundant bounding box
            for j in range(i+1, len(boxes)):
                boxj = boxes[j]
                if box_iou(boxi, boxj) >= 0.4:
                    boxes[j].probs[c] = 0.0
    return boxes

def process_box(self, b, h, w, threshold):
    max_indx = np.argmax(b.probs)
    max_prob = b.probs[max_indx]
    label = self.meta['labels'][max_indx]
    if max_prob > threshold:
        left = int((b.x - b.w/2.) * w)
        right = int((b.x + b.w/2.) * w)
        top = int((b.y - b.h/2.) * h)
        bottom = int((b.y + b.h/2.) * h)
        if left  < 0    :  left = 0
        if right > w - 1: right = w - 1
        if top   < 0    :   top = 0
        if bottom   > h - 1:   bottom = h - 1
        mess = '{}'.format(label)
        return (left, right, top, bottom, mess, max_indx, max_prob)
    return None
    
def preprocess(self, im, allobj=None):
    if type(im) is not np.ndarray:
        im = cv2.imread(im)
    
    if allobj is not None: # in training mode
        result = imcv2_affine_trans(im)
        im, dims, trans_param = result
        scale, offs, flip = trans_param
        """
        for obj in allobj:
            _fix(obj, dims, scale, offs)
            if not flip: 
                continue
            obj_1_ = obj[1]
            obj[1] = dims[0] - obj[3]
            obj[3] = dims[0] - obj_1_
        """
        im = imcv2_recolor(im)
    # resize image to input size
    im = self.resize_input(im)
    return im
        
def postprocess(self, net_out, im, save=True):
    # compute bounding box of each class
    boxes = self.findboxes(net_out)
    meta = self.meta
    threshold = meta['thresh']
    colors = meta['colors']
    if type(im) is not np.ndarray:
        img = cv2.imread(im)
    else:
        img = im
    h, w, _ = img.shape
    result = list()
    for b in boxes:
        boxResults = self.process_box(b, h, w, threshold)
        if boxResults is None:
            continue
        left, right, top, bottom, mess, max_indx, confidence = boxResults
        # put result in list
        result.append(dict({'label':mess,
                            'confidence':confidence,
                            'xmin':left,
                            'xmax':right,
                            'ymin':top,
                            'ymax':bottom}))
        # save image
        thick = int((h + w) // 300)
        cv2.rectangle(img, (left, top), (right, bottom), 
                         colors[max_indx], thick)
        cv2.putText(img, mess, (left, top - 12),
                    0, 1e-3 * h, colors[max_indx], thick // 3)
    img_name = os.path.join(self.args['output'], im.split('\\')[-1])
    if self.args['save']:
        cv2.imwrite(img_name, img)
    return result

