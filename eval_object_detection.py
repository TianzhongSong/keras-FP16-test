# coding=utf8
from models.keras_ssd300 import ssd_300
from models.keras_ssd512 import ssd_512
from utils.object_detection_2d_data_generator import DataGenerator
from utils.coco_utils import get_coco_category_maps, predict_all_to_json
from utils.average_precision_evaluator import Evaluator
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from models.yolo3 import yolo_eval
from keras.models import load_model
import numpy as np
import argparse
import keras.backend as K

K.clear_session()

weights = {'ssd300voc': 'VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.h5',
           'ssd512voc': 'VGG_VOC0712Plus_SSD_512x512_iter_240000.h5',
           'ssd300coco': 'VGG_coco_SSD_300x300_iter_400000.h5',
           'ssd512coco': 'VGG_coco_SSD_512x512_iter_360000.h5',
           'yolov3': 'yolov3.h5'}


class Yolo(object):
    def __init__(self, input_shape=(320, 320), score=0.01, iou_threshold=0.45):
        self.input_shape = input_shape
        self.score = score
        self.iou = iou_threshold
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = './data/coco_classes.txt'
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = './data/yolo_anchors.txt'
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = './weights/{}'.format(weights['yolov3'])
        self.yolo_model = load_model(model_path, compile=False)
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def predict(self, inputs):
        outputs = []
        for item in inputs:
            item = np.float32(item)
            item /= 255.
            item = np.expand_dims(item, 0)
            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.yolo_model.input: item,
                    self.input_image_shape: [self.input_shape[1], self.input_shape[0]],
                    K.learning_phase(): 0
                })
            output = []
            for i, c in reversed(list(enumerate(out_classes))):
                output.append([c] + out_scores[i] + out_boxes[i])
            outputs.append(output)
        return outputs


def create_model(model_type='ssd300', dataset='voc2007', dtype='float32'):
    if model_type == 'ssd300':
        model = ssd_300(image_size=(300, 300, 3),
                        n_classes=20 if dataset == 'voc2007' else 80,
                        mode='inference',
                        l2_regularization=0.0005,
                        scales=[0.1, 0.2, 0.37, 0.54,
                                0.71, 0.88, 1.05] if dataset == 'voc2007' else [0.07, 0.15, 0.33,
                                                                         0.51, 0.69, 0.87, 1.05],
                        aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                 [1.0, 2.0, 0.5],
                                                 [1.0, 2.0, 0.5]],
                        two_boxes_for_ar1=True,
                        steps=[8, 16, 32, 64, 100, 300],
                        offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                        clip_boxes=False,
                        variances=[0.1, 0.1, 0.2, 0.2],
                        normalize_coords=True,
                        subtract_mean=[123, 117, 104],
                        swap_channels=[2, 1, 0],
                        confidence_thresh=0.01,
                        iou_threshold=0.45,
                        top_k=200,
                        nms_max_output_size=400,
                        dtype=dtype)
        if dataset == 'voc2007':
            model.load_weights('./weights/{}'.format(weights['ssd300voc']), by_name=True)
        else:
            model.load_weights('./weights/{}'.format(weights['ssd300coco']), by_name=True)
    elif model_type == 'ssd512':
        model = ssd_512(image_size=(512, 512, 3),
                        n_classes=20 if dataset == 'voc2007' else 80,
                        mode='inference',
                        l2_regularization=0.0005,
                        scales=[0.07, 0.15, 0.3, 0.45,
                                0.6, 0.75, 0.9, 1.05] if dataset == 'voc2007' else [0.04, 0.1, 0.26,
                                                                                0.42, 0.58, 0.74, 0.9, 1.06],
                        aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                                 [1.0, 2.0, 0.5],
                                                 [1.0, 2.0, 0.5]],
                        two_boxes_for_ar1=True,
                        steps=[8, 16, 32, 64, 128, 256, 512],
                        offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                        clip_boxes=False,
                        variances=[0.1, 0.1, 0.2, 0.2],
                        normalize_coords=True,
                        subtract_mean=[123, 117, 104],
                        swap_channels=[2, 1, 0],
                        confidence_thresh=0.01,
                        iou_threshold=0.45,
                        top_k=200,
                        nms_max_output_size=400,
                        dtype=dtype)
        if dataset == 'voc2007':
            model.load_weights('./weights/{}'.format(weights['ssd512voc']), by_name=True)
        else:
            model.load_weights('./weights/{}'.format(weights['ssd512coco']), by_name=True)
    elif model_type in ['yolo320', 'yolo416', 'yolo608']:
        img_height = int(model_type[-3:])
        img_width = img_height
        model = Yolo(input_shape=(img_height, img_width),
                     score=0.01,
                     iou_threshold=0.45)
    else:
        raise ValueError("Only support SSD300, SSD512 and YOLOv3 now!")

    return model


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--model', type=str, default='ssd300', help='supports ssd300, ssd512, yolo320, yolo416, yolo608')
    parse.add_argument('--dtype', type=str, default='float16')
    parse.add_argument('--eval-dataset', type=str, default='coco', help='supports voc2007, coco')
    args = parse.parse_args()

    K.set_floatx(args.dtype)

    if args.eval_dataset == 'voc2007':
        assert args.model not in ['yolo320', 'yolo416', 'yolo608']

    model = create_model(model_type=args.model, dataset=args.eval_dataset, dtype=args.dtype)
    if args.eval_dataset == 'voc2007':
        img_height = int(args.model[-3:])
        img_width = img_height
        n_classes = 20

        Pascal_VOC_dataset_images_dir = '../../datasets/VOCdevkit/VOC2007/JPEGImages/'
        Pascal_VOC_dataset_annotations_dir = '../../datasets/VOCdevkit/VOC2007/Annotations/'
        Pascal_VOC_dataset_image_set_filename = '../../datasets/VOCdevkit/VOC2007/ImageSets/Main/test.txt'

        classes = ['background',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat',
                   'chair', 'cow', 'diningtable', 'dog',
                   'horse', 'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor']
        dataset = DataGenerator()
        dataset.parse_xml(images_dirs=[Pascal_VOC_dataset_images_dir],
                          image_set_filenames=[Pascal_VOC_dataset_image_set_filename],
                          annotations_dirs=[Pascal_VOC_dataset_annotations_dir],
                          classes=classes,
                          include_classes='all',
                          exclude_truncated=False,
                          exclude_difficult=False,
                          ret=False)
        evaluator = Evaluator(model=model,
                              n_classes=n_classes,
                              data_generator=dataset)

        results = evaluator(img_height=img_height,
                            img_width=img_width,
                            batch_size=8,
                            data_generator_mode='resize',
                            round_confidences=False,
                            matching_iou_threshold=0.5,
                            border_pixels='include',
                            sorting_algorithm='quicksort',
                            average_precision_mode='sample',
                            num_recall_points=11,
                            ignore_neutral_boxes=True,
                            return_precisions=True,
                            return_recalls=True,
                            return_average_precisions=True,
                            verbose=True)
        mean_average_precision, average_precisions, precisions, recalls = results
        print('Evaluating {0} with {1}'.format(args.model, args.eval_dataset))

        for i in range(1, len(average_precisions)):
            print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))
        print()
        print("{:<14}{:<6}{}".format('', 'mAP', round(mean_average_precision, 3)))

    elif args.eval_dataset == 'coco':
        img_height = int(args.model[-3:])
        img_width = img_height
        n_classes = 80

        dataset = DataGenerator()

        # Set the paths to the dataset here.
        MS_COCO_dataset_images_dir = '../../datasets/val2017/'
        MS_COCO_dataset_annotations_filename = '../../datasets/annotations/instances_val2017.json'

        dataset.parse_json(images_dirs=[MS_COCO_dataset_images_dir],
                           annotations_filenames=[MS_COCO_dataset_annotations_filename],
                           ground_truth_available=False,
                           include_classes='all',
                           ret=False)

        cats_to_classes, classes_to_cats, cats_to_names, classes_to_names = get_coco_category_maps(
            MS_COCO_dataset_annotations_filename)

        results_file = 'detections_val2017_ssd300_results.json'
        batch_size = 20
        predict_all_to_json(out_file=results_file,
                            model=model,
                            img_height=img_height,
                            img_width=img_width,
                            classes_to_cats=classes_to_cats,
                            data_generator=dataset,
                            batch_size=batch_size,
                            data_generator_mode='resize',
                            confidence_thresh=0.01,
                            iou_threshold=0.45,
                            top_k=200,
                            normalize_coords=True,
                            mode=args.model)
        coco_gt = COCO(MS_COCO_dataset_annotations_filename)
        coco_dt = coco_gt.loadRes(results_file)
        image_ids = sorted(coco_gt.getImgIds())

        cocoEval = COCOeval(cocoGt=coco_gt,
                            cocoDt=coco_dt,
                            iouType='bbox')
        cocoEval.params.imgIds = image_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
    else:
        raise ValueError('Only support VOC2007 and COCO!')
