# Pytorch implementation of Faster RCNN

An easy pytorch implementaion of Faster RCNN.
The dataset is images with color ellipse on black background. 
The goal is to detect the bounding rectangle of the ellipse.
The images are generated with OPENCV on the fly. 


## How to use

1. Install Python >= 3.6

2. Install necessary packages
```bash
    pip install -r requirements
```

3. Train a Faster-RCNN model. Here we alternatively train each part of the model:

    * Train **Backbone/RPN**
    * Train **Backbone/Fast RCNN Head**
    * Fine tune **RPN**
    * Fine tune **Fast RCNN Head**

   When training **Fast RCNN Head**, we use the previously trained **Backbone/RPN** to 
   deliver region proposals. Please refer to the [original paper](https://arxiv.org/abs/1506.01497) for more info. The above steps are included in one command line:
```bash
    python train.py
```

4. Now we have generated three files: 
    * *save_bbn.weights*
    * *save_rpn.weights*
    * *save_fhd.weights*

   We combine the networks to detect ellipse.
```bash
    python detect.py
```

## Prediction Results

Training takes around three hours on our PC:

| CPU | I5 9400F |
| --- | --- |
| GPU | GTX-1050Ti |
| RAM | 8GB |

With the trained weights, the model can predict bounding boxes of ellipses. We are presenting two examples, and the value attached to the top left corner is the confidence of the prediction:

![predict_1.jpg](https://wx2.sinaimg.cn/small/008b8Ivhgy1ghwgs9bidzj30hd0hd74n.jpg)
![predict_2.jpg](https://wx4.sinaimg.cn/small/008b8Ivhgy1ghwgscxaonj30hd0hdjrq.jpg)

