# coding=utf8
from models.keras_ssd300 import ssd_300
from keras.optimizers import Adam
from utils.keras_ssd_loss import SSDLoss
from utils.object_detection_2d_data_generator import DataGenerator
from utils.average_precision_evaluator import Evaluator
import keras.backend as K
import argparse

K.clear_session()


def create_model(model_type='ssd300', dataset='voc2007', dtype='float32'):
    if model_type == 'ssd300':
        model = ssd_300(image_size=(300, 300, 3),
                        n_classes=20 if dataset == 'voc2007' else 80,
                        mode='inference',
                        l2_regularization=0.0005,
                        # scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05], # coco
                        scales=[0.1, 0.2, 0.37, 0.54,
                                0.71, 0.88, 1.05] if dataset == 'voc' else [0.07, 0.15, 0.33,
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
            model.load_weights('./weights/ssd.h5', by_name=True)
        else:
            model.load_weights('./weights/ssd_coco.h5', by_name=True)
    elif model_type == 'yolov3':
        pass
    else:
        raise ValueError("Only support SSD300 and YOLOv3 now!")

    return model


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--model', type=str, default='ssd300')
    parse.add_argument('--dtype', type=str, default='float32')
    parse.add_argument('--eval-dataset', type=str, default='voc2007')
    args = parse.parse_args()

    K.set_floatx(args.dtype)

    if args.mdoel == 'ssd300':
        img_height, img_width = 300, 300
        n_classes = 20
        model = create_model(model_type=args.model, dataset=args.eval_dataset, dtype=args.dtype)
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
        model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

        dataset = DataGenerator()

        Pascal_VOC_dataset_images_dir = '../../datasets/VOCdevkit/VOC2007/JPEGImages/'
        Pascal_VOC_dataset_annotations_dir = '../../datasets/VOCdevkit/VOC2007/Annotations/'
        Pascal_VOC_dataset_image_set_filename = '../../datasets/VOCdevkit/VOC2007/ImageSets/Main/test.txt'

        classes = ['background',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat',
                   'chair', 'cow', 'diningtable', 'dog',
                   'horse', 'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor']

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

        for i in range(1, len(average_precisions)):
            print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))
        print()
        print("{:<14}{:<6}{}".format('', 'mAP', round(mean_average_precision, 3)))
