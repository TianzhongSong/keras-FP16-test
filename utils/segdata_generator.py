import numpy as np
import cv2
import random
import os
import sys

sys.path.append('../')


def get_batch(items, root_path, nClasses, height, width, train=True):
    x = []
    y = []
    for item in items:
        image_path = root_path + item.split(' ')[0]
        label_path = root_path + item.split(' ')[-1].strip()
        img = cv2.imread(image_path, 1)
        label_img = cv2.imread(label_path, 1)
        # random flip images during training
        if train:
            is_flip = random.randint(0, 1)
            if is_flip == 1:
                img = cv2.flip(img, 1)
                label_img = cv2.flip(label_img, 1)
        label_img = label_img[:, :, 0]
        seg_labels = np.zeros((height, width, nClasses))
        for c in range(nClasses):
            seg_labels[:, :, c] = (label_img == c).astype(int)
        img = np.float32(img) / 127.5 - 1
        seg_labels = np.reshape(seg_labels, (width * height, nClasses))
        x.append(img)
        y.append(seg_labels)
    return x, y


def generator(root_path, path_file, batch_size, n_classes, input_height, input_width):
    items = os.listdir(path_file)

    while True:
        shuffled_items = []
        index = [n for n in range(len(items))]
        random.shuffle(index)
        for i in range(len(items)):
            shuffled_items.append(items[index[i]])
        for j in range(len(items) // batch_size):
            x, y = get_batch(shuffled_items[j * batch_size:(j + 1) * batch_size],
                             root_path, n_classes, input_height, input_width)
            yield np.array(x), np.array(y)
