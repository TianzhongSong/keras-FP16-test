## Keras float16 vs float32

### Requirements

keras==2.1.0

tensorflow==1.8.0

opencv==3.2.0

pycocotools, BeautifulSoup4, lxml, tqdm

## Image Classification Part

### Usage

Firstly, download [ImageNet val data](http://ml.cs.tsinghua.edu.cn/~chenxi/dataset/val224_compressed.pkl)  and model pre-trained weights file.

An example for testing vgg16 with float16.

    python eval_image_classification.py --model='vgg' --dtype='float16'

An example for testing mobilenet with a width multiplier 1.0.

    python eval_image_classification.py --model='mobilenet' --dtype='float16' --alpha=1.0
    
### ImageNet Datatset

[ImageNet val data](http://ml.cs.tsinghua.edu.cn/~chenxi/dataset/val224_compressed.pkl) 
provided by [aaron-xichen](https://github.com/aaron-xichen), 
sincerely thanks to aaron-xichen for sharing this processed ImageNet val data.

### Results

TOP1 acc and TOP5 acc on ImageNet val data.

Pre-trained weight files are downloaded from [deep learning models](https://github.com/fchollet/deep-learning-models), 
DenseNet121 weight file is downloaded from [DenseNet-Keras](https://github.com/flyyufelix/DenseNet-Keras)
Squeezenet weight file is downloaded from [keras-squeezenet](https://github.com/rcmalli/keras-squeezenet)

There is a backup of weights in [baiduyun（百度云）](https://pan.baidu.com/s/17PHfl34pdiqPu0-87MqQvQ)

<table width="95%">
  <tr>
    <td></td>
    <td colspan=2 align=center>float32</td>
    <td colspan=2 align=center>float16</td>
    <td colspan=2 align=center>diff</td>
  </tr>
  <tr>
    <td align=center><b>Model</td>
    <td align=center>Top1 acc</td>
    <td align=center>Top5 acc</td>
    <td align=center>Top1 acc</td>
    <td align=center>Top5 acc</td>
    <td align=center>Top1 acc</td>
    <td align=center>Top5 acc</td>
  </tr>
  <tr>
    <td align=center width="10%"><b><a href="https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5">VGG16</a></td>
    <td align=center width="10%"><b>0.70786</td>
    <td align=center width="10%"><b>0.89794</td>
    <td align=center width="10%"><b>0.7082</td>
    <td align=center width="10%"><b>0.89802</td>
    <td align=center width="10%"><b>0.00034</td>
    <td align=center width="10%"><b>0.00008</td>
  </tr>
  <tr>
    <td align=center width="10%"><b><a href="https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5">ResNet50</a></td>
    <td align=center width="10%"><b>0.74366</td>
    <td align=center width="10%"><b>0.91806</td>
    <td align=center width="10%"><b>0.70508</td>
    <td align=center width="10%"><b>0.89466</td>
    <td align=center width="10%"><b>-0.03858</td>
    <td align=center width="10%"><b>-0.0234</td>
  </tr>
  <tr>
    <td align=center width="10%"><b><a href="https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5">Inceptionv3</a></td>
    <td align=center width="10%"><b>0.76518</td>
    <td align=center width="10%"><b>0.92854</td>
    <td align=center width="10%"><b>0.765</td>
    <td align=center width="10%"><b>0.92854</td>
    <td align=center width="10%"><b>-0.00018</td>
    <td align=center width="10%"><b>0.00</td>
  </tr>
  <tr>
    <td align=center width="10%"><b><a href="https://github.com/fchollet/deep-learning-models/releases/download/v0.7/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5">Inception-ResNet</a></td>
    <td align=center width="10%"><b>0.789</td>
    <td align=center width="10%"><b>0.94426</td>
    <td align=center width="10%"><b>0.7888</td>
    <td align=center width="10%"><b>0.94436</td>
    <td align=center width="10%"><b>-0.0002</td>
    <td align=center width="10%"><b>0.0001</td>
  </tr>
  <tr>
    <td align=center width="10%"><b><a href="https://drive.google.com/open?id=0Byy2AcGyEVxfSTA4SHJVOHNuTXc">DenseNet121</a></td>
    <td align=center width="10%"><b>0.74234</td>
    <td align=center width="10%"><b>0.91868</td>
    <td align=center width="10%"><b>0.74206</td>
    <td align=center width="10%"><b>0.91868</td>
    <td align=center width="10%"><b>-0.00028</td>
    <td align=center width="10%"><b>0.000</td>
  </tr>
  <tr>
    <td align=center width="10%"><b><a href="https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5">Xception</a></td>
    <td align=center width="10%"><b>0.77446</td>
    <td align=center width="10%"><b>0.93618</td>
    <td align=center width="10%"><b>0.77392</td>
    <td align=center width="10%"><b>0.93596</td>
    <td align=center width="10%"><b>-0.00054</td>
    <td align=center width="10%"><b>-0.00049</td>
  </tr>
  <tr>
    <td align=center width="10%"><b><a href="https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels.h5">Squeezenet</a></td>
    <td align=center width="10%"><b>0.52294</td>
    <td align=center width="10%"><b>0.76312</td>
    <td align=center width="10%"><b>0.52172</td>
    <td align=center width="10%"><b>0.76226</td>
    <td align=center width="10%"><b>-0.00122</td>
    <td align=center width="10%"><b>-0.00086</td>
  </tr>
  <tr>
    <td align=center width="10%"><b><a href="https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_1_0_224_tf.h5">MobileNet-1-0</a></td>
    <td align=center width="10%"><b>0.69856</td>
    <td align=center width="10%"><b>0.89174</td>
    <td align=center width="10%"><b>0.6966</td>
    <td align=center width="10%"><b>0.8898</td>
    <td align=center width="10%"><b>-0.00196</td>
    <td align=center width="10%"><b>-0.00194</td>
  </tr>
  <tr>
    <td align=center width="10%"><b><a href="https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_7_5_224_tf.h5">MobileNet-7-5</a></td>
    <td align=center width="10%"><b>0.67726</td>
    <td align=center width="10%"><b>0.87838</td>
    <td align=center width="10%"><b>0.6726</td>
    <td align=center width="10%"><b>0.87652</td>
    <td align=center width="10%"><b>-0.00466</td>
    <td align=center width="10%"><b>-0.00186</td>
    </tr>
  <tr>
    <td align=center width="10%"><b><a href="https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_5_0_224_tf.h5">MobileNet-5-0</a></td>
    <td align=center width="10%"><b>0.6352</td>
    <td align=center width="10%"><b>0.85006</td>
    <td align=center width="10%"><b>0.62944</td>
    <td align=center width="10%"><b>0.84644</td>
    <td align=center width="10%"><b>-0.00576</td>
    <td align=center width="10%"><b>-0.00362</td>
  </tr>
  <tr>
    <td align=center width="10%"><b><a href="https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_2_5_224_tf.h5">MobileNet-2-5</a></td>
    <td align=center width="10%"><b>0.5134</td>
    <td align=center width="10%"><b>0.75546</td>
    <td align=center width="10%"><b>0.50272</td>
    <td align=center width="10%"><b>0.74648</td>
    <td align=center width="10%"><b>-0.01068</td>
    <td align=center width="10%"><b>-0.00898</td>
  </tr>
</table>

## Object Detection Part

### Usage

Firstly, download [VOC2007 test set](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar) and [COCO2017 val set](http://images.cocodataset.org/zips/val2017.zip), [COCO2017 val set annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) datasets, then extract them and modify the path in [scripts](https://github.com/TianzhongSong/keras-FP16-test/blob/master/eval_object_detection.py).

Secondly, download SSD pre-trained weights and put them in 'weights' directory.

[SSD300 VOC](https://drive.google.com/file/d/1fyDDUcIOSjeiP08vl1WCndcFdtboFXua/view), [SSD300 COCO](https://drive.google.com/open?id=1vmEF7FUsWfHquXyCqO17UaXOPpRbwsdj), [SSD512 VOC](https://drive.google.com/file/d/18nFnqv9fG5Rh_fx6vUtOoQHOLySt4fEx/view), [SSD512 COCO](https://drive.google.com/open?id=1IJWZKmjkcFMlvaz2gYukzFx4d6mH3py5)

An example for evaluating SSD300 on VOC2007 test set

    python eval_object_detection.py --model='ssd300' --dtype='float16' --eval-dataset='voc2007'

### Results

SSD300 result on VOC2007 test set

<table width="95%">
  <tr>
    <td></td>
    <td colspan=3 align=center>mAP</td>
  </tr>
  <tr>
    <td align=center><b>Model</td>
    <td align=center>float32</td>
    <td align=center>float16</td>
    <td align=center>diff</td>
  </tr>
  <tr>
    <td align=center width="10%"><b>SSD300</td>
    <td align=center width="10%"><b>0.782</td>
    <td align=center width="10%"><b>0.769</td>
    <td align=center width="10%"><b>-0.013</td>
  </tr>
</table>

The AP of each category can be found in this [doc](https://github.com/TianzhongSong/keras-FP16-test/blob/master/docs/ssd_results.md)

### ToDo

Object detection(SSD, Yolo)

Semantic segmentation

### Reference

[deep learing models](https://github.com/fchollet/deep-learning-models)

[DenseNet-Keras](https://github.com/flyyufelix/DenseNet-Keras)

[keras-squeezenet](https://github.com/rcmalli/keras-squeezenet)

[ssd_keras](https://github.com/pierluigiferrari/ssd_keras)

[keras-yolo3](https://github.com/qqwweee/keras-yolo3)
