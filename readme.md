## Keras float16 vs float32

### Requirements

keras==2.1.0

tensorflow==1.8.0

opencv==3.2.0

tqdm

## Image Classification Part

### Usage

Firstly, download [ImageNet val data](http://ml.cs.tsinghua.edu.cn/~chenxi/dataset/val224_compressed.pkl)  and model pre-trained weights file.

An example for testing vgg16 with float16.

    python run_image_classification.py --model='vgg' --weights='vgg16_weights.h5' --dtype='float16'

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

|Model                  | float32              |float16                 |diff                  |
| :-------------------: |:--------------------:|:---------------------: |:-----------------------:|
|[VGG16](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5)                 | 0.70786/0.89794      | 0.7082/0.89802         | 0.00034/0.00008   |
|[ResNet50](https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5)               | 0.74366/0.91806      | 0.70508/0.89466        | -0.03858/-0.0234    |
|[Inceptionv3](https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5)            | 0.76518/0.92854      | 0.765/0.92854          | -0.00018/0.00    |
|[Inception-ResNet](https://github.com/fchollet/deep-learning-models/releases/download/v0.7/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5)       | 0.789/0.94426        | 0.7888/0.94436         | -0.0002/0.0001    |
|[DenseNet121](https://drive.google.com/open?id=0Byy2AcGyEVxfSTA4SHJVOHNuTXc)            | 0.74234/0.91868      | 0.74206/0.91868        | -0.00028/0.000     |
|[Xception](https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5)               | 0.77446/0.93618      | 0.77392/0.93596        |  -0.00054/-0.00049    |
|[Squeezenet](https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels.h5)             | 0.52294/0.76312      | 0.52172/0.76226        |   -0.00122/-0.00086     |
|[MobileNet-1-0](https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_1_0_224_tf.h5)          | 0.69856/0.89174      | 0.6966/0.8898          |   -0.00196/-0.00194    |
|[MobileNet-7-5](https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_7_5_224_tf.h5)          | 0.67726/0.87838      | 0.6726/0.87652         |   -0.00466/-0.00186    |
|[MobileNet-5-0](https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_5_0_224_tf.h5)          | 0.6352/0.85006       | 0.62944/0.84644        |   -0.00576/-0.00362   |
|[MobileNet-2-5](https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_2_5_224_tf.h5)          | 0.5134/0.75546       | 0.50272/0.74648         |  -0.01068/-0.00898   |

### ToDo

Object detection

Semantic segmentation

### Reference

[deep learing models](https://github.com/fchollet/deep-learning-models)

[DenseNet-Keras](https://github.com/flyyufelix/DenseNet-Keras)

[keras-squeezenet](https://github.com/rcmalli/keras-squeezenet)

