import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from dataset.dataloader import load_tfds
from time import time
import argparse
from nets.simple_basline import SimpleBaseline
from nets.evopose2d import EvoPose
from nets.hrnet import HRNet
from utils import detect_hardware


def speed_test(strategy, cfg, split='val', n=1000):
    with strategy.scope():
        if cfg.MODEL.TYPE == 'simple_baseline':
            model = SimpleBaseline(cfg)
        elif cfg.MODEL.TYPE == 'hrnet':
            model = HRNet(cfg)
        elif cfg.MODEL.TYPE == 'evopose':
            model = EvoPose(cfg)

    cfg.DATASET.OUTPUT_SHAPE = model.output_shape[1:]

    ds = load_tfds(cfg, split, det=cfg.VAL.DET,
                   predict_kp=True, drop_remainder=cfg.VAL.DROP_REMAINDER)
    ds = strategy.experimental_distribute_dataset(ds)

    @tf.function
    def predict(imgs, flip=False):
        if flip:
            imgs = imgs[:, :, ::-1, :]
        return model(imgs, training=False)

    for count, batch in enumerate(ds):
        if count == 1:  # skip first pass
            ti = time()

        _, imgs, _, _, scores = batch

        hms = strategy.run(predict, args=(imgs,)).numpy()

        if cfg.VAL.FLIP:
            flip_hms = strategy.run(predict, args=(imgs, True,)).numpy()
            flip_hms = flip_hms[:, :, ::-1, :]
            tmp = flip_hms.copy()
            for i in range(len(cfg.DATASET.KP_FLIP)):
                flip_hms[:, :, :, i] = tmp[:, :, :, cfg.DATASET.KP_FLIP[i]]
            # shift to align features
            flip_hms[:, :, 1:, :] = flip_hms[:, :, 0:-1, :].copy()
            hms = (hms + flip_hms) / 2.

        if count == n:
            break

    print('FPS: {:.5f}'.format((n * cfg.VAL.BATCH_SIZE) / (time() - ti)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--tpu', default='')
    parser.add_argument('-c', '--cfg', required=True)  # yaml
    parser.add_argument('-bs', '--batch-size', type=int, default=1)
    parser.add_argument('-n', type=int, default=1000)
    args = parser.parse_args()

    from dataset.coco import cn as cfg
    cfg.merge_from_file('configs/' + args.cfg)
    cfg.MODEL.NAME = args.cfg.split('.')[0]
    cfg.VAL.BATCH_SIZE = args.batch_size

    if args.cpu:
        strategy = tf.distribute.OneDeviceStrategy('/CPU:0')
    elif args.gpu:
        strategy = tf.distribute.OneDeviceStrategy('/GPU:0')
    else:
        tpu, strategy = detect_hardware(args.tpu)

    tf.config.optimizer.set_experimental_options({'disable_meta_optimizer': True})
    speed_test(strategy, cfg, split='val', n=args.n)



