# -*- coding: utf-8 -*-
from . import data
from . import test
from . import train
import numpy as np
from os import sep

class YOLOv2(object):
    parse = data.parse
    shuffle = data.shuffle
    preprocess = test.preprocess
    postprocess = test.postprocess
    _batch = data._batch
    resize_input = test.resize_input
    findboxes = test.findboxes
    process_box = test.process_box
    build_loss = train.build_loss
    
    def __init__(self, meta, args):
        model = meta['model'].split(sep)[-1]
        model = '.'.join(model.split('.')[:-1])
        meta['name'] = model
        self.constructor(meta, args)
    
    # set plot color and label
    def constructor(self, meta, args):
        def _to_color(indx, base):
            base2 = base * base
            b = 2 - indx / base2
            r = 2 - (indx % base2) / base
            g = 2 - (indx % base2) % base
            return (b*127, r*127, g*127)
                
        # set label
        with open(args['label_file'], 'r') as f:
            meta['labels'] = list()
            labs = [l.strip() for l in f.readlines()]
            for lab in labs:
                if lab == '----': break
                meta['labels'] += [lab]
        #print meta['labels']
        # check
        #print(len(meta['labels']), meta['classes'])
        assert len(meta['labels']) == meta['classes'], \
                'labels.txt and {} indicate' + ' ' \
                'inconsistent class numbers'.format(meta['model'])
        # plot color set
        colors = list()
        base = int(np.ceil(pow(meta['classes'], 1./3)))
        for x in range(len(meta['labels'])):
            colors += [_to_color(x, base)]
        meta['colors'] = colors
        self.fetch = list()
        self.meta, self.args = meta, args
    