import tensorflow as tf
import os.path as osp
import math
from tensorflow.python.keras.layers.preprocessing import image_preprocessing as image_ops
import cv2
import numpy as np


def parse_record(record):
    feature_description = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'img_id': tf.io.FixedLenFeature([], tf.int64),
        'bbox': tf.io.FixedLenFeature([4, ], tf.float32),
        'joints': tf.io.FixedLenFeature([51, ], tf.int64),
        'score': tf.io.FixedLenFeature([], tf.float32)
    }
    example = tf.io.parse_single_example(record, feature_description)
    img_id = example['img_id']
    img = tf.image.decode_jpeg(example['image_raw'], channels=3)
    bbox = example['bbox']
    kp = tf.reshape(example['joints'], [-1, 3])
    score = example['score']
    return img_id, img, bbox, kp, score


def transform(img, scale, angle, bbox_center, output_shape):
    tx = bbox_center[0] - output_shape[1] * scale / 2
    ty = bbox_center[1] - output_shape[0] * scale / 2

    # for offsetting translations caused by rotation:
    # https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html
    rx = (1 - tf.cos(angle)) * output_shape[1] * scale / 2 - tf.sin(angle) * output_shape[0] * scale / 2
    ry = tf.sin(angle) * output_shape[1] * scale / 2 + (1 - tf.cos(angle)) * output_shape[0] * scale / 2

    transform = [scale * tf.cos(angle), scale * tf.sin(angle), rx + tx,
                 -scale * tf.sin(angle), scale * tf.cos(angle), ry + ty,
                 0., 0.]

    img = image_ops.transform(tf.expand_dims(img, axis=0),
                              tf.expand_dims(transform, axis=0),
                              fill_mode='constant',
                              output_shape=output_shape[:2])
    img = tf.squeeze(img)

    # transform for keypoints
    alpha = 1 / scale * tf.cos(-angle)
    beta = 1 / scale * tf.sin(-angle)

    rx_xy = (1 - alpha) * bbox_center[0] - beta * bbox_center[1]
    ry_xy = beta * bbox_center[0] + (1 - alpha) * bbox_center[1]

    transform_xy = [[alpha, beta],
                    [-beta, alpha]]

    tx_xy = bbox_center[0] - output_shape[1] / 2
    ty_xy = bbox_center[1] - output_shape[0] / 2

    M = tf.concat([transform_xy, [[rx_xy - tx_xy], [ry_xy - ty_xy]]], axis=1)
    return img, M


def preprocess(id, img, bbox, kp, score, DATASET, split='train', predict_kp=False):
    img = tf.cast(img, tf.float32)
    kp = tf.cast(kp, tf.float32)

    if DATASET.BGR:
        img = tf.gather(img, [2, 1, 0], axis=-1)

    if DATASET.NORM:
        img /= 255.
        if DATASET.BGR:
            img -= [[DATASET.MEANS[::-1]]]
            img /= [[DATASET.STDS[::-1]]]
        else:
            img -= [[DATASET.MEANS]]
            img /= [[DATASET.STDS]]

    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    center = [x + w / 2., y + h / 2.]
    aspect_ratio = DATASET.INPUT_SHAPE[1] / DATASET.INPUT_SHAPE[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    scale = (h * 1.25) / DATASET.INPUT_SHAPE[0]

    if split == 'train':
        # augmentation
        scale *= tf.clip_by_value(tf.random.normal([]) * DATASET.SCALE_FACTOR + 1,
                                  1 - DATASET.SCALE_FACTOR,
                                  1 + DATASET.SCALE_FACTOR)

        if tf.random.uniform([]) <= DATASET.ROT_PROB:
            angle = tf.clip_by_value(tf.random.normal([]) * DATASET.ROT_FACTOR,
                                      -2*DATASET.ROT_FACTOR, 2*DATASET.ROT_FACTOR)/180*math.pi
        else:
            angle = 0.

        if tf.random.uniform([]) <= DATASET.FLIP_PROB:
            img_w = tf.cast(tf.shape(img)[1], tf.float32)
            img = img[:, ::-1, :]
            center_x = img_w - 1 - center[0]
            kp_x = img_w - 1 - kp[:, 0]
            kp = tf.concat([tf.expand_dims(kp_x, axis=1), kp[:, 1:]], axis=-1)
            kp = tf.gather(kp, DATASET.KP_FLIP, axis=0)
            center = [center_x, center[1]]
    else:
        angle = 0.

    img, M = transform(img, scale, angle, center, DATASET.INPUT_SHAPE[:2])

    xy = kp[:, :2]
    xy = tf.transpose(tf.matmul(M[:, :2], xy, transpose_b=True)) + M[:, -1]

    # adjust visibility if coordinates are outside crop
    vis = kp[:, 2]
    vis *= tf.cast((
            (xy[:, 0] >= 0) &
            (xy[:, 0] < DATASET.INPUT_SHAPE[1]) &
            (xy[:, 1] >= 0) &
            (xy[:, 1] < DATASET.INPUT_SHAPE[0])), tf.float32)

    kp = tf.concat([xy, tf.expand_dims(vis, axis=1)], axis=1)

    if DATASET.BFLOAT16:
        img = tf.cast(img, tf.bfloat16)

    if predict_kp:
        return id, img, kp, M, score
    else:
        return img, kp


def generate_heatmaps(kp, DATASET):
    x = [i for i in range(DATASET.OUTPUT_SHAPE[1])]
    y = [i for i in range(DATASET.OUTPUT_SHAPE[0])]
    xx, yy = tf.meshgrid(x, y)
    xx = tf.reshape(tf.dtypes.cast(xx, tf.float32), (1, *DATASET.OUTPUT_SHAPE[:2], 1))
    yy = tf.reshape(tf.dtypes.cast(yy, tf.float32), (1, *DATASET.OUTPUT_SHAPE[:2], 1))

    x = tf.floor(tf.reshape(kp[:, :, 0], [-1, 1, 1, DATASET.OUTPUT_SHAPE[-1]])
                 / DATASET.INPUT_SHAPE[1] * DATASET.OUTPUT_SHAPE[1] + 0.5)
    y = tf.floor(tf.reshape(kp[:, :, 1], [-1, 1, 1, DATASET.OUTPUT_SHAPE[-1]])
                 / DATASET.INPUT_SHAPE[0] * DATASET.OUTPUT_SHAPE[0] + 0.5)

    heatmaps = tf.exp(-(((xx - x) / DATASET.SIGMA) ** 2) / 2 - (
                ((yy - y) / DATASET.SIGMA) ** 2) / 2) * 255.

    valid = tf.cast(kp[:, :, -1] > 0, tf.float32)
    return heatmaps, valid


def load_tfds(cfg, split, det=False, predict_kp=False, drop_remainder=True):
    record_subdir = osp.join(cfg.DATASET.TFRECORDS, split)
    if split == 'val' and det:
        record_subdir = osp.join(record_subdir, 'dets')
    file_pattern = osp.join(record_subdir, '*.tfrecord')
    AUTO = tf.data.experimental.AUTOTUNE
    ds = tf.data.Dataset.list_files(file_pattern, shuffle=True)
    ds = ds.interleave(tf.data.TFRecordDataset,
                       cycle_length=10,
                       block_length=1,
                       num_parallel_calls=AUTO)
    ds = ds.cache() if cfg.DATASET.CACHE else ds
    ds = ds.shuffle(10000).repeat() if split == 'train' else ds
    ds = ds.map(lambda record: parse_record(record), num_parallel_calls=AUTO)
    ds = ds.map(lambda id, img, bbox, kp, score:
                preprocess(id, img, bbox, kp, score, cfg.DATASET, split, predict_kp),
                num_parallel_calls=AUTO)
    if split == 'train':
        ds = ds.batch(cfg.TRAIN.BATCH_SIZE).prefetch(AUTO)
    else:
        ds = ds.batch(cfg.VAL.BATCH_SIZE, drop_remainder=drop_remainder).prefetch(AUTO)
    if not predict_kp:
        ds = ds.map(lambda imgs, kp: (imgs, *generate_heatmaps(kp, cfg.DATASET)),
                    num_parallel_calls=AUTO)
    return ds


def visualize(img, joints, valid):
    for i, v in enumerate(valid):
        if v == 1:  # occluded
            cv2.circle(img, tuple(joints[i]), 1, (0, 255, 0))
        elif v == 2:  # visible
            cv2.circle(img, tuple(joints[i]), 1, (0, 0, 255))
    return img


if __name__ == '__main__':
    tf.random.set_seed(0)

    from coco import cfg
    cfg.DATASET.INPUT_SHAPE = [512, 388, 3]
    cfg.DATASET.NORM = False
    cfg.DATASET.BGR = True

    ds = load_tfds(cfg, 'val', det=False, predict_kp=True)
    for id, img, kp, M, score in ds:
        cv2.imshow('', visualize(np.uint8(img[0]), kp[0, :, :2].numpy(), kp[0, :, -1].numpy()))
        cv2.waitKey()
        cv2.destroyAllWindows()

