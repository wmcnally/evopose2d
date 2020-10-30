import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import os.path as osp
import pickle
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tensorflow as tf
import numpy as np
from dataset.dataloader import load_tfds
from tqdm import tqdm
import math
import json
import cv2
from utils import get_flops, detect_hardware
import sys
from tensorflow.keras.mixed_precision import experimental as mixed_precision


def get_preds(hms, Ms, input_shape, output_shape):
    preds = np.zeros((hms.shape[0], output_shape[-1], 3))
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            hm = hms[i, :, :, j]
            idx = hm.argmax()
            y, x = np.unravel_index(idx, hm.shape)
            px = int(math.floor(x + 0.5))
            py = int(math.floor(y + 0.5))
            if 1 < px < output_shape[1] - 1 and 1 < py < output_shape[0] - 1:
                diff = np.array([hm[py][px + 1] - hm[py][px - 1],
                                 hm[py + 1][px] - hm[py - 1][px]])
                diff = np.sign(diff)
                x += diff[0] * 0.25
                y += diff[1] * 0.25
            preds[i, j, :2] = [x * input_shape[1] / output_shape[1],
                              y * input_shape[0] / output_shape[0]]
            preds[i, j, -1] = hm.max() / 255

    # use inverse transform to map kp back to original image
    for j in range(preds.shape[0]):
        M_inv = cv2.invertAffineTransform(Ms[j])
        preds[j, :, :2] = np.matmul(M_inv[:, :2], preds[j, :, :2].T).T + M_inv[:, 2].T
    return preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tpu', default=None)
    parser.add_argument('-m', '--model-name', required=True)
    parser.add_argument('--coco-path', default='/media/wmcnally/data/coco')
    parser.add_argument('-bs', '--batch-size', type=int, default=256)
    parser.add_argument('--det', type=int, default=0)
    parser.add_argument('--score-thresh', type=float, default=0.2)
    args = parser.parse_args()

    meta_data = pickle.load(open('models/' + args.model_name.split('.h5')[0] + '_meta.pkl', 'rb'))
    cfg = meta_data['config']

    # if cfg.DATASET.BFLOAT16:
    #     policy = mixed_precision.Policy('mixed_float16')
    #     mixed_precision.set_policy(policy)

    cfg.VAL.BATCH_SIZE = args.batch_size
    cfg.DATASET.CACHE = False
    # cfg.DATASET.BFLOAT16 = False
    # sys.exit()

    tpu, strategy = detect_hardware(args.tpu)

    if tpu:
        cfg.DATASET.TFRECORDS = 'gs://willmcnally1/TF2-SimpleHumanPose/tfrecords'  # public bucket
        drop_remainder = True  # this is required with TPU, will reduce AP
    else:
        cfg.DATASET.TFRECORDS = '/media/wmcnally/data/coco/TF2-SimpleHumanPose/tfrecords'
        drop_remainder = False

    result_path = 'models/{}-result.json'.format(args.model_name.split('.h5')[0])
    coco = COCO(osp.join(args.coco_path, 'annotations', 'person_keypoints_val2017.json'))

    with strategy.scope():
        model = tf.keras.models.load_model('models/' + args.model_name, compile=False)

    print('Loaded checkpoint {}'.format(args.model_name))
    print('Parameters: {:.2f}M'.format(meta_data['parameters'] / 1e6))
    print('Multiply-Adds: {:.2f}G'.format(meta_data['flops'] / 2 / 1e9))

    @tf.function
    def predict(imgs, flip=False):
        if flip:
            imgs = imgs[:, :, ::-1, :]
        return model(imgs, training=False)

    ds = load_tfds(cfg, 'val', det=args.det, predict_kp=True, drop_remainder=drop_remainder)
    ds = strategy.experimental_distribute_dataset(ds)

    results = []
    for batch in tqdm(ds):
        ids, imgs, _, Ms, scores = batch

        ids = np.concatenate(ids.values, axis=0)
        scores = np.concatenate(scores.values, axis=0)
        Ms = np.concatenate(Ms.values, axis=0)

        hms = strategy.run(predict, args=(imgs,)).values
        hms = np.concatenate(hms, axis=0)

        if cfg.VAL.FLIP:
            flip_hms = strategy.run(predict, args=(imgs, True,)).values
            flip_hms = np.concatenate(flip_hms, axis=0)
            flip_hms = flip_hms[:, :, ::-1, :]
            tmp = flip_hms.copy()
            for i in range(len(cfg.DATASET.KP_FLIP)):
                flip_hms[:, :, :, i] = tmp[:, :, :, cfg.DATASET.KP_FLIP[i]]
            # shift to align features
            flip_hms[:, :, 1:, :] = flip_hms[:, :, 0:-1, :].copy()
            hms = (hms + flip_hms) / 2.

        preds = get_preds(hms, Ms, cfg.DATASET.INPUT_SHAPE, cfg.DATASET.OUTPUT_SHAPE)
        kp_scores = preds[:, :, -1].copy()

        # rescore
        rescored_score = np.zeros((len(kp_scores)))
        for i in range(len(kp_scores)):
            score_mask = kp_scores[i] > args.score_thresh
            if np.sum(score_mask) > 0:
                rescored_score[i] = np.mean(kp_scores[i][score_mask]) * scores[i]
        score_result = rescored_score

        for j in range(preds.shape[0]):
            results.append(dict(image_id=int(ids[j]),
                                category_id=1,
                                keypoints=preds[j].reshape(-1).tolist(),
                                score=float(score_result[j])))

    with open(result_path, 'w') as f:
        json.dump(results, f)

    result = coco.loadRes(result_path)
    cocoEval = COCOeval(coco, result, iouType='keypoints')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()