# -*- coding: utf-8 -*-

def parser(model):
    def _parse(l, i=1):
        return l.split('=')[i].strip()

    with open(model, 'rb') as f:
        lines = f.readlines()

    lines = [line.decode() for line in lines]

    meta = dict()
    layers = list()  # will contains layers' info
    h, w, c = [0] * 3
    layer = dict()
    for indice, line in enumerate(lines):
        line = line.split('#')[0]
        line = line.strip()
        if line == '': continue
        if '[' in line:
            if layer != dict():
                if layer['type'] == '[net]':
                    h = layer['height']
                    w = layer['width']
                    c = layer['channels']
                    meta['net'] = layer
                else:
                    if layer['type'] == '[crop]':
                        h = layer['crop_height']
                        w = layer['crop_width']
                    layers += [layer]
            layer = {'type': line}
        else:
            try:
                i = float(_parse(line))
                if i == int(i): i = int(i)
                layer[line.split('=')[0].strip()] = i
            except:
                try:
                    key = _parse(line, 0)
                    val = _parse(line, 1)
                    layer[key] = val
                except:
                    print('parse configure file error, in {0} lines: {1}.'.format(
                        indice + 1, line))

    meta.update(layer)  # last layer contains meta info
    if 'anchors' in meta:
        splits = meta['anchors'].split(',')
        anchors = [float(x.strip()) for x in splits]
        meta['anchors'] = anchors
    meta['model'] = model  # path to cfg, not model name
    meta['inp_size'] = [h, w, c]
    return layers, meta


def cfg_yielder(cfg_file):
    layers, meta = parser(cfg_file)
    yield meta
    h, w, c = meta['inp_size']
    l = w * h * c

    # Start yielding
    flat = False  # flag for 1st dense layer
    for i, d in enumerate(layers):
        #-----------------------------------------------------
        if d['type'] == '[convolutional]':
            n = d.get('filters', 1)
            size = d.get('size', 1)
            stride = d.get('stride', 1)
            pad = d.get('pad', 0)
            padding = d.get('padding', 0)
            if pad: padding = size // 2
            activation = d.get('activation', 'logistic')
            batch_norm = d.get('batch_normalize', 0)
            yield [
                'convolutional', i, size, c, n, stride, padding, batch_norm,
                activation
            ]
            if activation != 'linear': yield [activation, i]
            w_ = (w + 2 * padding - size) // stride + 1
            h_ = (h + 2 * padding - size) // stride + 1
            w, h, c = w_, h_, n
            l = w * h * c
        #-----------------------------------------------------
        elif d['type'] == '[maxpool]':
            stride = d.get('stride', 1)
            size = d.get('size', stride)
            padding = d.get('padding', (size - 1) // 2)
            yield ['maxpool', i, size, stride, padding]
            w_ = (w + 2 * padding) // d['stride']
            h_ = (h + 2 * padding) // d['stride']
            w, h = w_, h_
            l = w * h * c
        #-----------------------------------------------------
        elif d['type'] == '[route]':  # add new layer here
            routes = d['layers']
            if type(routes) is int:
                routes = [routes]
            else:
                routes = [int(x.strip()) for x in routes.split(',')]
            routes = [i + x if x < 0 else x for x in routes]
            for j, x in enumerate(routes):
                lx = layers[x]
                _size = lx['_size'][:3]
                if not j: w, h, c = _size
                else: 
                    w_, h_, c_ = _size
                    assert w_ == w and h_ == h, 'Routing incompatible conv sizes'
                    c += c_
            yield ['route', i, routes]
            l = w * h * c
        #-----------------------------------------------------
        elif d['type'] == '[reorg]':
            stride = d.get('stride', 1)
            yield ['reorg', i, stride]
            w = w // stride
            h = h // stride
            c = c * (stride**2)
            l = w * h * c
        #-----------------------------------------------------
        else:
            exit('Layer {} not implemented'.format(d['type']))

        d['_size'] = list([h, w, c, l, flat])

    if not flat:
        meta['out_size'] = [h, w, c]
    else:
        meta['out_size'] = l
