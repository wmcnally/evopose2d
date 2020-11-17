import os
import os.path as osp
from multiprocessing import get_context
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from train import train
import argparse
from dataset.coco import cn as cfg
from utils import get_models
import copy
from nets.evopose2d import genotype_from_blocks_args, DEFAULT_BLOCKS_ARGS, mutate
from time import sleep
import numpy as np
import pickle


def update_cfg(cfg, accelerator, gen, model_id,
               genotype, epochs, parent, disp):
    cfg.MODEL.SAVE_DIR = osp.join(cfg.SEARCH.DIR, '{:03d}'.format(gen))
    cfg.MODEL.NAME = '{:03d}_{:05d}'.format(gen, model_id)
    cfg.MODEL.GENOTYPE = list(genotype)
    cfg.MODEL.PARENT = parent
    cfg.TRAIN.ACCELERATOR = accelerator
    cfg.TRAIN.EPOCHS = epochs
    cfg.TRAIN.DISP = disp
    return cfg


def train_generation(cfg, accelerators, meta_files, models, genotypes):
    gens = [np.int(m.split('_')[0]) for m in meta_files]
    completed_ids = [np.int(m.split('_')[1]) for m in meta_files]
    train_cycles = cfg.SEARCH.CHILDREN // len(accelerators)

    if np.sum(gens == np.max(gens)) >= cfg.SEARCH.CHILDREN:
        gen = np.max(gens) + 1
        print('Training generation {}'.format(gen))
        print('Performing {} train cycles of {} children'
              .format(train_cycles, cfg.SEARCH.CHILDREN // train_cycles))
    elif len(gens) == 1:
        gen = 1
        print('Training generation {}'.format(gen))
        print('Performing {} train cycles of {} children'
              .format(train_cycles, cfg.SEARCH.CHILDREN // train_cycles))
    else:
        gen = np.max(gens)
        print('Continuing generation', gen)
        models = [m for m in models if m.split('_')[0] != '{:03d}'.format(gen)]

    last_model_id = np.max([np.int(m.split('_')[1])
                            for m in meta_files
                            if m.split('_')[0] != '{:03d}'.format(gen)])

    if len(models) == 1:
        parents = [models[0] for _ in range(cfg.SEARCH.CHILDREN)]
        parent_genotypes = [genotypes[0] for _ in range(cfg.SEARCH.CHILDREN)]
    else:
        fitness = [np.float32(m.split('_')[-1].split('.h5')[0]) for m in models]
        parent_idx = np.argsort(fitness)[:cfg.SEARCH.PARENTS]  # mu models with best fitness (lowest loss)
        parents = [models[i] for i in parent_idx]
        print('Parents: {}'.format(parents))
        print('Cleaning up models that are not parents...')
        not_parents = [m for m in models if m not in parents and np.int(m.split('_')[0]) != 0]
        for m in not_parents:
            os.remove(osp.join(cfg.SEARCH.DIR, m.split('_')[0], m))
        parents *= cfg.SEARCH.CHILDREN // cfg.SEARCH.PARENTS
        parent_genotypes = [genotypes[np.int(p.split('_')[1])] for p in parents]
    parents = [osp.join(cfg.SEARCH.DIR, p.split('_')[0], p) for p in parents]  # paths

    for i in range(train_cycles):
        train_cfgs = []
        for j in range(len(accelerators)):
            model_id = i * len(accelerators) + j + last_model_id + 1
            if model_id not in completed_ids:
                np.random.seed(model_id + cfg.TRAIN.SEED)
                genotype = mutate(parent_genotypes[i * len(accelerators) + j],
                                  cache=[np.array(genotypes[i]) for i in genotypes.keys()])
                genotypes[model_id] = genotype
                parent = parents[i * len(accelerators) + j]
                train_cfgs.append(
                    update_cfg(copy.deepcopy(cfg), accelerators[j], gen, model_id,
                            genotype, cfg.SEARCH.EPOCHS, parent, False))
        if len(train_cfgs) > 0:
            with get_context("spawn").Pool(len(accelerators)) as p:
                p.map(train_wrapper, train_cfgs)


def train_wrapper(cfg):
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=cfg.TRAIN.ACCELERATOR)  # TPU detection
    except:
        tpu = None
    if tpu:
        tpu_init = False
        while not tpu_init:
            try:
                tf.config.experimental_connect_to_cluster(tpu)
                tf.tpu.experimental.initialize_tpu_system(tpu)
                tpu_init = True
            except:
                print('Could not connect to {}. Waiting 5 seconds and trying again...'
                      .format(cfg.TRAIN.ACCELERATOR))
                sleep(5)
        strategy = tf.distribute.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.OneDeviceStrategy(cfg.TRAIN.ACCELERATOR)

    model, meta_data = train(strategy, cfg)
    meta_data['fitness'] = min(meta_data['val_loss']) * (cfg.SEARCH.TARGET / meta_data['parameters']) ** cfg.SEARCH.W

    print('{} ({:.2f}M / {:.2f}G) loss: {:.4f} - fit: {:.4f} - {:.2f} mins'
          .format(cfg.MODEL.NAME,
                  meta_data['parameters'] / 1e6,
                  meta_data['flops'] / 2 / 1e9,
                  min(meta_data['val_loss']),
                  meta_data['fitness'],
                  meta_data['training_time'] / 60))

    cfg.MODEL.NAME += '_{:.5f}'.format(meta_data['fitness'])
    model.save(osp.join(cfg.MODEL.SAVE_DIR, '{}.h5'.format(cfg.MODEL.NAME)), save_format='h5')
    pickle.dump(meta_data, open(osp.join(cfg.MODEL.SAVE_DIR,
                                         '{}_meta.pkl'.format(cfg.MODEL.NAME)), 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='E3.yaml')
    parser.add_argument('--accelerator-prefix', default='node-')
    parser.add_argument('-a', '--accelerator-ids', type=int, nargs='+', default=[0])
    parser.add_argument('-ar', '--accelerator-range', type=int, nargs='+', default=None)
    args = parser.parse_args()

    cfg.merge_from_file(osp.join('configs/', args.cfg))

    if args.accelerator_range:
        accelerators = ['{}{}'.format(args.accelerator_prefix, a)
                        for a in range(*args.accelerator_range)]
    else:
        accelerators = ['{}{}'.format(args.accelerator_prefix, a)
                        for a in args.accelerator_ids]

    assert cfg.SEARCH.CHILDREN % cfg.SEARCH.PARENTS == 0, \
        'Number of children must be divisible by number of parents.'
    assert cfg.SEARCH.CHILDREN % len(accelerators) == 0, \
        'Number of children must be divisible by number of accelerators.'

    print('Using accelerators: {}'.format(accelerators))

    cfg.SEARCH.DIR = 'searches/' + args.cfg.split('.')[0]
    os.makedirs(cfg.SEARCH.DIR, exist_ok=True)

    # EVOLUTION
    while 1:  # each loop is a generation, run until manually stopped
        meta_files, models, genotypes = get_models(cfg.SEARCH.DIR)

        if len(models) == 0:
            # train initial genotype
            train_cfg = update_cfg(copy.deepcopy(cfg),
                                   accelerator=accelerators[0],
                                   gen=0, model_id=0,
                                   genotype=genotype_from_blocks_args(DEFAULT_BLOCKS_ARGS),
                                   epochs=cfg.SEARCH.GEN0_EPOCHS, parent=None, disp=True)
            train_wrapper(train_cfg)
        else:
            # train next generation
            train_generation(copy.deepcopy(cfg), accelerators, meta_files, models, genotypes)