from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152
from tensorflow.keras.regularizers import l2

from utils import add_regularization, get_flops

BACKBONES = {
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'resnet152': ResNet152
}


def SimpleBaseline(cfg):
    regularizer = l2(cfg.TRAIN.WD)

    if cfg.MODEL.LOAD_WEIGHTS:
        weights = 'imagenet'
    else:
        weights = None

    backbone = BACKBONES[cfg.MODEL.BACKBONE](
        weights=weights,
        include_top=False,
        input_shape=cfg.DATASET.INPUT_SHAPE)

    backbone = add_regularization(backbone, regularizer)

    x = backbone.output
    for i in range(3):
        x = layers.Conv2DTranspose(
            cfg.MODEL.HEAD_CHANNELS,
            cfg.MODEL.HEAD_KERNEL,
            strides=2,
            padding='same',
            use_bias=False,
            kernel_regularizer=regularizer,
            name='head_conv{}'.format(i + 1))(x)
        x = layers.BatchNormalization(name='head_bn{}'.format(i + 1))(x)
        x = layers.Activation(cfg.MODEL.HEAD_ACTIVATION, name='head_act{}'.format(i + 1))(x)

    x = layers.Conv2D(
        cfg.DATASET.OUTPUT_SHAPE[-1],
        1,
        padding='same',
        use_bias=True,
        kernel_regularizer=regularizer,
        name='final_conv')(x)

    return Model(backbone.input, x, name='sb_{}'.format(cfg.MODEL.BACKBONE))


if __name__ == '__main__':
    from dataset.coco import cn as cfg
    cfg.merge_from_file('../configs/sb_resnet50_256x192.yaml')
    cfg.DATASET.INPUT_SHAPE = [384, 288, 3]
    cfg.MODEL.BACKBONE = 'resnet152'
    model = SimpleBaseline(cfg)
    model.summary()
    print('{:.2f}M parameters | {:.2f}G multiply-adds'
          .format(model.count_params() / 1e6, get_flops(model) / 1e9 / 2))
