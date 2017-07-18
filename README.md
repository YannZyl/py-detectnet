# py-detectnet-master

Yolov2 is a good object detect framework. It has been introduced in research [YOLO](https://pjreddie.com/media/files/papers/yolo.pdf), [YOLOv2](https://arxiv.org/abs/1612.08242), [GitHub](https://github.com/thtrieu/darkflow)

There is a image detect using yolov2 framework

![image](https://github.com/YannZyl/py-detectnet-master/blob/master/output/2008_004040.jpg '2008_004040.jpg')

# Demo

Using yolov2 to detect object, just modify **main.py**
```bash
args = dict({
    # network toplogy and pre-training model
    'cfg_file': 'files/yolo_yann.cfg',
    'model_file': 'files/yolo_yann_89000.weights',
    # test images directory
    'test': 'test',
    # output images directory
    'output': 'output',
})
```

And then run **main.py** with
```bash
python main.py
```

# Train

To accelated train state, we use darknet(c++) to train our model, please see [here](https://pjreddie.com/darknet/). You should design your own model and after training, you can perform our framework to detect image, because it is more flexible than c++ version. 



