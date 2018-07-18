import numpy as np
import cv2
import itertools
import os
import sys

sys.path.append('../')


def getImageArr(path, width, height, imgNorm="sub_mean"):
    try:
        img = cv2.imread(path, 1)
        if imgNorm == "sub_and_divide":
            img = np.float32(img) / 127.5 -1
        elif imgNorm == "sub_mean":
            img = cv2.resize(img, (width, height))
            img = img.astype(np.float32)
            img[:, :, 0] -= 103.939
            img[:, :, 1] -= 116.779
            img[:, :, 2] -= 123.68
        elif imgNorm == "divide":
            img = cv2.resize(img, (width, height))
            img = img.astype(np.float32)
            img = img / 255.0
        return img
    except Exception as e:
        print(path, e)
        img = np.zeros((height, width, 3))
        return img


def getSegmentationArr(path, nClasses, width, height):
    seg_labels = np.zeros((height, width, nClasses))
    try:
        img = cv2.imread(path, 1)
        img = cv2.resize(img, (width, height))
        img = img[:, :, 0]

        for c in range(nClasses):
            seg_labels[:, :, c] = (img == c).astype(int)

    except Exception as e:
        print(e)

    seg_labels = np.reshape(seg_labels, (width * height, nClasses))
    return seg_labels


def imageSegmentationGenerator(images_path, segs_path, batch_size, n_classes, input_height, input_width, output_height,
                               output_width):
    images = os.listdir(images_path)
    segmentations = os.listdir(segs_path)
    images.sort()
    segmentations.sort()
    for i in range(len(images)):
        images[i] = images_path + images[i]
        segmentations[i] = segs_path + segmentations[i]

    assert len(images) == len(segmentations)
    for im, seg in zip(images, segmentations):
        im = im.replace('\\', '/')
        seg = seg.replace('\\', '/')
        assert (im.split('/')[-1].split(".")[0] == seg.split('/')[-1].split(".")[0])

    zipped = itertools.cycle(zip(images, segmentations))

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            im, seg = next(zipped)
            X.append(getImageArr(im, input_width, input_height))
            Y.append(getSegmentationArr(seg, n_classes, output_width, output_height))

        yield np.array(X), np.array(Y)
