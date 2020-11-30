from pycocotools.coco import COCO
import argparse
import os
import os.path as osp
import numpy as np
import json
import tensorflow as tf
import sys


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte (list)."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """Returns a float_list from a float / double (list)."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint (list)."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def tf_example(sample_dict):
    features = {}
    image_raw = open(sample_dict['img_path'], 'rb').read()
    features['image_raw'] = _bytes_feature([image_raw])
    features['bbox'] = _float_feature(sample_dict['bbox'])
    features['img_id'] = _int64_feature([sample_dict['img_id']])
    features['joints'] = _int64_feature(sample_dict['joints'])
    features['score'] = _float_feature([sample_dict['score']])
    return tf.train.Example(features=tf.train.Features(feature=features))


def load_data(annot_path, det_path=None, split='train'):

    coco = COCO(annot_path)
    coco_path = '/'.join(annot_path.split('/')[:-2])

    data = []

    if det_path is None:
        for aid in coco.anns.keys():
            ann = coco.anns[aid]
            joints = ann['keypoints']
            if split == 'train':
                if (ann['image_id'] not in coco.imgs) or ann['iscrowd'] or (np.sum(joints[2::3]) == 0) or (
                        ann['num_keypoints'] == 0):
                    continue
            else:
                if ann['image_id'] not in coco.imgs:
                    continue
            img_name = '{}2017/'.format(split) + coco.imgs[ann['image_id']]['file_name']

            # sanitize bboxes
            x, y, w, h = ann['bbox']
            img = coco.loadImgs(ann['image_id'])[0]
            width, height = img['width'], img['height']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if ann['area'] > 0 and x2 >= x1 and y2 >= y1:
                bbox = [x1, y1, x2 - x1, y2 - y1]
            else:
                continue

            d = dict(img_id=ann['image_id'],
                     img_path=osp.join(coco_path, 'images', img_name),
                     bbox=bbox,
                     joints=joints,
                     score=1)
            data.append(d)
    else:
        dets = json.load(open(det_path))
        for det in dets:
            if det['image_id'] not in coco.imgs or det['category_id'] != 1:
                continue
            img_name = '{}2017/'.format(split) + coco.imgs[det['image_id']]['file_name']
            d = dict(img_id=det['image_id'],
                     img_path=osp.join(coco_path, 'images', img_name),
                     bbox=det['bbox'],
                     joints=[0 for _ in range(17*3)],
                     score=det['score'])
            data.append(d)

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco-path', default='/media/user/data/coco')
    parser.add_argument('--write-dir', default='/media/user/data/coco/tfrecords')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'])
    parser.add_argument('--shard-size', type=int, default=1024)
    parser.add_argument('--dets', default=None)
    args = parser.parse_args()

    annot_dict = {
        'train': osp.join(args.coco_path, 'annotations/person_keypoints_train2017.json'),
        'val': osp.join(args.coco_path, 'annotations/person_keypoints_val2017.json'),
        'test': osp.join(args.coco_path, 'annotations/image_info_test-dev2017.json'),
    }

    for split in args.splits:
        write_subdir = osp.join(args.write_dir, split)
        if args.dets is not None:
            assert split in ['val', 'test'], "Split must be in ['val', 'test'] if using dets"
            if split == 'val':
                write_subdir = osp.join(write_subdir, 'dets')

        data = load_data(annot_dict[split], det_path=args.dets, split=split)
        os.makedirs(write_subdir, exist_ok=True)

        i = 0
        shard_count = 0
        while i < len(data):
            record_path = osp.join(write_subdir, '{:05d}.tfrecord'.format(shard_count))
            with tf.io.TFRecordWriter(record_path) as writer:
                for j in range(args.shard_size):
                    if args.dets:
                        if i >= len(data):
                            # write extra samples to fill the shard so we don't have to
                            # drop any samples when testing on TPU
                            example = tf_example(data[-1])
                        else:
                            example = tf_example(data[i])
                        writer.write(example.SerializeToString())
                        i += 1
                    else:
                        example = tf_example(data[i])
                        writer.write(example.SerializeToString())
                        i += 1
                        if i == len(data):
                            break
                if i >= len(data):
                    break
            print('Finished writing', record_path)
            shard_count += 1

