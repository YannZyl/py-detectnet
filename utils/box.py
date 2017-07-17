import numpy as np

class BoundBox(object):
    def __init__(self, classes):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.c = float()
        self.probs = np.zeros((classes,))
    

def box_iou(box1, box2):
    # boxes params
    xmin1, ymin1 = box1.x-box1.w/2., box1.y-box1.h/2.
    xmax1, ymax1 = box1.x+box1.w/2., box1.y+box1.h/2. 
    xmin2, ymin2 = box2.x-box2.w/2., box2.y-box2.h/2.
    xmax2, ymax2 = box2.x+box2.w/2., box2.y+box2.h/2.
    # overlap region coordinate
    x_lt = max(xmin1, xmin2)
    y_lt = max(ymin1, ymin2)
    x_rb = min(xmax1, xmax2)
    y_rb = min(ymax1, ymax2)
    # box area, intersection area and union area
    box1_area = box1.w * box1.h
    box2_area = box2.w * box2.h
    intersection = max(0.0, x_rb-x_lt) * max(0.0, y_rb-y_lt)
    union = box1_area + box2_area - intersection
    # compute iou
    return intersection / union
    