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
from dataset.coco import cn as cfg


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


def validate(strategy, cfg):
    cfg.DATASET.CACHE = False
    meta_data = pickle.load(open('models/' + cfg.MODEL.NAME + '_meta.pkl', 'rb'))
    result_path = 'models/{}-result.json'.format(cfg.MODEL.NAME)
    coco = COCO(cfg.DATASET.ANNOT)

    with strategy.scope():
        model = tf.keras.models.load_model('models/' + cfg.MODEL.NAME + '.h5', compile=False)

    print('Loaded checkpoint {}'.format(cfg.MODEL.NAME))
    print('Parameters: {:.2f}M'.format(meta_data['parameters'] / 1e6))
    print('Multiply-Adds: {:.2f}G'.format(meta_data['flops'] / 2 / 1e9))

    ds = load_tfds(cfg, 'val', det=cfg.VAL.DET,
                   predict_kp=True, drop_remainder=cfg.VAL.DROP_REMAINDER)
    ds = strategy.experimental_distribute_dataset(ds)

    @tf.function
    def predict(imgs, flip=False):
        if flip:
            imgs = imgs[:, :, ::-1, :]
        return model(imgs, training=False)

    results = []
    count = 0
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

        if count == 0:
            all_ids = ids
            all_hms = hms
            all_Ms = Ms
            all_scores = scores
        else:
            all_ids = np.concatenate((all_ids, ids), axis=0)
            all_hms = np.concatenate((all_hms, hms), axis=0)
            all_Ms = np.concatenate((all_Ms, Ms), axis=0)
            all_scores = np.concatenate((all_scores, scores), axis=0)
        count += 1

    ids = all_ids
    hms = all_hms
    Ms = all_Ms
    scores = all_scores

    preds = get_preds(hms, Ms, cfg.DATASET.INPUT_SHAPE, cfg.DATASET.OUTPUT_SHAPE)
    kp_scores = preds[:, :, -1].copy()

    # rescore
    rescored_score = np.zeros((len(kp_scores)))
    for i in range(len(kp_scores)):
        score_mask = kp_scores[i] > cfg.VAL.SCORE_THRESH
        if np.sum(score_mask) > 0:
            rescored_score[i] = np.mean(kp_scores[i][score_mask]) * scores[i]
    score_result = rescored_score

    for i in range(preds.shape[0]):
        results.append(dict(image_id=int(ids[i]),
                            category_id=1,
                            keypoints=preds[i].reshape(-1).tolist(),
                            score=float(score_result[i])))

    with open(result_path, 'w') as f:
        json.dump(results, f)

    result = coco.loadRes(result_path)
    cocoEval = COCOeval(coco, result, iouType='keypoints')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tpu', default=None)
    parser.add_argument('-c', '--cfg', required=True)  # yaml
    parser.add_argument('-bs', '--batch-size', type=int, default=None)
    parser.add_argument('--det', type=int, default=0)
    args = parser.parse_args()

    cfg.merge_from_file('configs/' + args.cfg)
    if args.batch_size:
        cfg.VAL.BATCH_SIZE = args.batch_size
    cfg.VAL.DET = True if args.det else False

    tpu, strategy = detect_hardware(args.tpu)
    validate(strategy, cfg)