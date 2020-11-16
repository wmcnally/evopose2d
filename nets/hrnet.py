from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2

from utils import add_regularization


def basic_block(x, out_channels, stride=1, name=''):
    res = x

    x = Conv2D(out_channels, 3, stride, 'same', use_bias=False,
               name=name + '/conv1')(x)
    x = BatchNormalization(name=name + '/bn1')(x)
    x = ReLU(name=name + '/relu1')(x)

    x = Conv2D(out_channels, 3, 1, 'same', use_bias=False,
               name=name + '/conv2')(x)
    x = BatchNormalization(name=name + '/bn2')(x)

    if stride != 1 or res.shape[-1] != x.shape[-1]:
        res = Conv2D(out_channels, 1, stride, use_bias=False,
                     name=name + '/skip_conv')(res)
        res = BatchNormalization(name=name + '/skip_bn')(res)

    x = Add(name=name + '/add')([x, res])
    x = ReLU(name=name + '/relu2')(x)
    return x


def bottleneck(x, out_channels, stride=1, expansion=4, name=''):
    res = x

    x = Conv2D(out_channels, 1, 1, use_bias=False,
               name=name + '/conv1')(x)
    x = BatchNormalization(name=name + '/bn1')(x)
    x = ReLU(name=name + '/relu1')(x)

    x = Conv2D(out_channels, 3, stride, 'same', use_bias=False,
               name=name + '/conv2')(x)
    x = BatchNormalization(name=name + '/bn2')(x)
    x = ReLU(name=name + '/relu2')(x)

    x = Conv2D(out_channels * expansion, 1, 1, use_bias=False,
               name=name + '/conv3')(x)
    x = BatchNormalization(name=name + '/bn3')(x)

    if stride != 1 or res.shape[-1] != x.shape[-1]:
        res = Conv2D(out_channels * expansion, 1, stride, use_bias=False,
                     name=name + '/skip_conv')(res)
        res = BatchNormalization(name=name + '/skip_bn')(res)

    x = Add(name=name + '/add')([x, res])
    x = ReLU(name=name + '/relu3')(x)
    return x


blocks_dict = {
    'basic': basic_block,
    'bottleneck': bottleneck,
}


def transition(xs, out_channels, name=''):
    n_branch_pre = len(xs)
    n_branch_cur = len(out_channels)

    xs_next = []
    for i in range(n_branch_cur):
        if i < n_branch_pre:
            x = xs[i]
            if x.shape[-1] != out_channels[i]:
                x = Conv2D(out_channels[i], 3, 1, 'same', use_bias=False,
                           name=name + '/b{}/conv'.format(i + 1))(x)
                x = BatchNormalization(name=name + '/b{}/bn'.format(i + 1))(x)
                x = ReLU(name=name + '/b{}/relu'.format(i + 1))(x)
        else:
            # new branch
            x = xs[-1]
            x = Conv2D(out_channels[i], 3, 2, 'same', use_bias=False,
                       name=name + '/b{}/conv'.format(i + 1))(x)
            x = BatchNormalization(name=name + '/b{}/bn'.format(i + 1))(x)
            x = ReLU(name=name + '/b{}/relu'.format(i + 1))(x)
        xs_next.append(x)
    return xs_next


def fuse_scales(xs, name='', upsample='transpose'):
    if len(xs) == 1:
        return xs

    fusion_outs = []
    for i in range(len(xs)):
        to_be_fused = []
        for j in range(len(xs)):
            x = xs[j]
            if j > i:
                if upsample == 'transpose':
                    x = Conv2DTranspose(xs[i].shape[-1], 1, 2 ** (j - i), use_bias=False,
                                        name=name + '/b{}{}_conv'.format(i, j))(x)
                    x = BatchNormalization(name=name + '/b{}{}_bn'.format(i, j))(x)
                else:
                    x = Conv2D(xs[i].shape[-1], 1, 1, use_bias=False,
                               name=name + '/b{}{}_conv'.format(i, j))(x)
                    x = BatchNormalization(name=name + '/b{}{}_bn'.format(i, j))(x)
                    x = UpSampling2D(size=2 ** (j - i), interpolation='nearest')(x)
            elif j < i:
                for k in range(i - j):
                    if k == i - j - 1:
                        out_channels = xs[i].shape[-1]
                        x = Conv2D(out_channels, 3, 2, 'same', use_bias=False,
                                   name=name + '/b{}{}_conv{}'.format(i, j, k))(x)
                        x = BatchNormalization(name=name + '/b{}{}_bn{}'.format(i, j, k))(x)
                    else:
                        out_channels = xs[j].shape[-1]
                        x = Conv2D(out_channels, 3, 2, 'same', use_bias=False,
                                   name=name + '/b{}{}_conv{}'.format(i, j, k))(x)
                        x = BatchNormalization(name=name + '/b{}{}_bn{}'.format(i, j, k))(x)
                        x = ReLU(name=name + '/b{}{}_relu{}'.format(i, j, k))(x)
            to_be_fused.append(x)
        x = Add(name=name + '/b{}_add'.format(i))(to_be_fused)
        x = ReLU(name=name + '/b{}_relu'.format(i))(x)
        fusion_outs.append(x)
    return fusion_outs


def module(xs, stage_cfg, name=''):
    for i in range(len(xs)):
        for j in range(stage_cfg['NUM_BLOCKS'][i]):
            xs[i] = blocks_dict[stage_cfg['BLOCK']](
                xs[i], stage_cfg['NUM_CHANNELS'][i],
                name=name + '/b{}{}'.format(i, j))
    xs = fuse_scales(xs, name=name + '/fusion', upsample=stage_cfg.UPSAMPLE)
    return xs


def stage(xs, stage_cfg, name=''):
    xs = transition(xs, stage_cfg['NUM_CHANNELS'], name=name + '/transition')
    for i in range(stage_cfg['NUM_MODULES']):
        xs = module(xs, stage_cfg, name=name + '/m{}'.format(i))
    return xs


def HRNet(cfg):
    input = Input(cfg.DATASET.INPUT_SHAPE)
    x = input

    # stem
    for i in range(2):
        x = Conv2D(64, 3, 2, 'same', use_bias=False,
                   name='conv{}'.format(i + 1))(x)
        x = BatchNormalization(name='bn{}'.format(i + 1))(x)
        x = ReLU(name='relu{}'.format(i + 1))(x)

    xs = [x]
    stage_cfgs = [cfg.MODEL['STAGE{}'.format(i+1)] for i in range(4)]
    for i, stage_cfg in enumerate(stage_cfgs):
        xs = stage(xs, stage_cfg, name='s{}'.format(i + 1))

    output = Conv2D(cfg.DATASET.OUTPUT_SHAPE[-1], 1, name='final_conv')(xs[0])
    model = Model(inputs=input, outputs=output, name='hrnet')
    add_regularization(model, l2(cfg.TRAIN.WD))
    return model


if __name__ == '__main__':
    from utils import get_flops
    from dataset.coco import cn as cfg
    cfg.merge_from_file('../configs/hrnet_w32_256x192.yaml')
    # cfg.DATASET.INPUT_SHAPE = [256, 192, 3]
    model = HRNet(cfg)
    # model.summary()
    print('{:.2f}M / {:.2f}G'.format(model.count_params() / 1e6, get_flops(model) / 1e9 / 2))