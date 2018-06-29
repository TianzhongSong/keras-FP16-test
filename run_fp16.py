import argparse
from models import vgg16, inception_v3, resnet_50, mobilenet, densenet_121, xception, inception_resnet_v2
from pkl_reader import DataGenerator
from keras import backend as K
import numpy as np

K.clear_session()


def top5_acc(pred, k=5):
    Inf = 0.
    results = []
    for i in range(k):
        results.append(pred.index(max(pred)))
        pred[pred.index(max(pred))] = Inf
    return results


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='command for testing keras model with fp16 and fp32')
    parse.add_argument('--model', type=str, default='xception', help='support vgg16, resnet50, densenet121, inceptionv3, inception_resnet_v2, xception, mobilenet')
    parse.add_argument('--weights', type=str, default='./weights/xception_weights_tf_dim_ordering_tf_kernels.h5')
    parse.add_argument('--dtype', type=str, default='float32')
    parse.add_argument('--alpha', type=float, default=0.25, help='alpha for mobilenet')
    args = parse.parse_args()

    K.set_floatx(args.dtype)

    # create model
    if args.model == 'vgg':
        model = vgg16.VGG16(input_shape=(224, 224, 3))
    elif args.model == 'inception':
        model = inception_v3.InceptionV3(input_shape=(299, 299, 3))
    elif args.model == 'inception_resnet':
        model = inception_resnet_v2.InceptionResNetV2(input_shape=(299, 299, 3))
    elif args.model == 'xception':
        model = xception.Xception(input_shape=(299, 299, 3))
    elif args.model == 'resnet':
        model = resnet_50.ResNet50(input_shape=(224, 224, 3))
    elif args.model == 'densenet':
        model = densenet_121.DenseNet(reduction=0.5, classes=1000)
    elif args.model == 'mobilenet':
        model = mobilenet.MobileNet(input_shape=(224, 224, 3), alpha=args.alpha)
    else:
        raise ValueError("Do not support {}".format(args.model))
    # model.summary()
    model.load_weights(args.weights)

    print('Evaluation {0} with {1} data type'.format(args.model, args.dtype))

    dg = DataGenerator('./imagenet_val/val224_compressed.pkl', model=args.model, dtype=args.dtype)
    acc = 0
    acc_top5 = 0
    for im, label in dg.generator():
        out = model.predict(im)
        pred = np.argmax(out[0])
        if pred == label:
            acc += 1
        if label in top5_acc(out[0].tolist()):
            acc_top5 += 1
    print('Top1 accuracy: {}'.format(acc / 50000))
    print('Top5 accuracy: {}'.format(acc_top5 / 50000))
