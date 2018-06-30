## Keras float16 vs float32

### Requirements

keras==2.0.8

tensorflow==1.3.0

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

Notice: DepthWiseConv dose not support float16 in tensorflow1.3.0

Pre-trained weights file download from [deep learning models](https://github.com/fchollet/deep-learning-models), 
DenseNet121 weights download from [DenseNet-Keras](https://github.com/flyyufelix/DenseNet-Keras)

There is a backup of weights in [baiduyun（百度云）](https://pan.baidu.com/s/1oIiDqQBVko39M9_C3RvFlQ)

|Model                  | float32              |float16                 |
| :-------------------: |:--------------------:|:---------------------: |
| VGG16                 | 0.70786/0.8979       | 0.70796/0.89786        |
|ResNet50               | 0.74366/0.91806      | 0.70508/0.89466        |
|Inceptionv3            | 0.76518/0.92854      | 0.765/0.92854          |
|Inception-ResNet       | 0.789/0.94426        | 0.7888/0.94436          |
|DenseNet121            | 0.74234/0.91868      | 0.74206/0.91868        |
|Xception               | 0.77446/0.93618      | fail                   |
|MobileNet-1-0          | 0.69856/0.89174      | fail |
|MobileNet-7-5          | 0.67726/0.87838      | fail |
|MobileNet-5-0          | 0.6352/0.85006       | fail |
|MobileNet-2-5          | 0.5134/0.75546       | fail |

### Reference

[deep learing models](https://github.com/fchollet/deep-learning-models)

[DenseNet-Keras](https://github.com/flyyufelix/DenseNet-Keras)
