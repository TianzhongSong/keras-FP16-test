## Keras float16 vs float32

### Requirements

keras==2.1.0

tensorflow==1.8.0

opencv==3.2.0

tqdm

### Usage

An example for testing vgg16 with float16.

    python run.py --model='vgg' --weights='vgg16_weights.h5' --dtype='float16'

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

|Model                  | float32              |float16                 |
| :-------------------: |:--------------------:|:---------------------: |
|VGG16                  | 0.70786/0.89794      | 0.7082/0.89802         |
|ResNet50               | 0.74366/0.91806      | 0.70508/0.89466        |
|Inceptionv3            | 0.76518/0.92854      | 0.765/0.92854          |
|Inception-ResNet       | 0.789/0.94426        | 0.7888/0.94436         |
|DenseNet121            | 0.74234/0.91868      | 0.74206/0.91868        |
|Xception               | 0.77446/0.93618      | 0.77392/0.93596        |
|Squeezenet             | 0.52294/0.76312      | 0.52172/0.76226        |
|MobileNet-1-0          | 0.69856/0.89174      | 0.6966/0.8898          |
|MobileNet-7-5          | 0.67726/0.87838      | 0.6726/0.87652         |
|MobileNet-5-0          | 0.6352/0.85006       | 0.62944/0.84644        |
|MobileNet-2-5          | 0.5134/0.75546       | 0.50272/074648         |

### Reference

[deep learing models](https://github.com/fchollet/deep-learning-models)

[DenseNet-Keras](https://github.com/flyyufelix/DenseNet-Keras)

[keras-squeezenet](https://github.com/rcmalli/keras-squeezenet)

