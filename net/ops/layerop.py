# -*- coding: utf-8 -*-
from .baseop import *
from .layers import *
op_type = {
    'convolutional': Convolutional,
    'maxpool': Maxpool,
    'leaky': Leaky,
    'identity': Identity,
    'route': Route,
    'reorg': Reorg
}

def create_layerop(*args):
    layer_type = list(args)[0].type
    return op_type[layer_type](*args)
