import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import copy
import math
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB4, EfficientNetB5
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l2
import numpy as np
from utils import partial_weight_transfer

DEFAULT_BLOCKS_ARGS = [{
    'kernel_size': 3,
    'repeats': 1,
    'filters_out': 16,
    'expand_ratio': 1,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 2,
    'filters_out': 24,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 2,
    'filters_out': 40,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 3,
    'filters_out': 80,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 3,
    'filters_out': 112,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 4,
    'filters_out': 192,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 1,
    'filters_out': 320,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}]

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


def get_parent(input_shape):
    if input_shape[0] == 256:
        parent = EfficientNetB0(include_top=False)
    elif input_shape[0] == 384:
        parent = EfficientNetB4(include_top=False)
    elif input_shape[0] == 512:
        parent = EfficientNetB5(include_top=False)
    else:
        print('Not a valid input shape for EfficientNet parent')
        parent = None
    return parent


def blocks_args_from_genotype(genotype):
    blocks_args = copy.deepcopy(DEFAULT_BLOCKS_ARGS)
    for i, args in enumerate(genotype):
        blocks_args[i]['kernel_size'] = int(args[0])
        blocks_args[i]['repeats'] = int(args[1])
        blocks_args[i]['filters_out'] = int(args[2]*8)
        blocks_args[i]['strides'] = int(args[3])
    return blocks_args


def genotype_from_blocks_args(blocks_args):
    genotype = []
    for args in copy.deepcopy(blocks_args):
        genotype.append([args['kernel_size'], args['repeats'], args['filters_out']//8, args['strides']])
    return genotype


def round_filters(filters, width_coefficient, divisor=8):
    """Round number of filters based on depth multiplier."""
    filters *= width_coefficient
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""
    return int(math.ceil(depth_coefficient * repeats))


def EvoPose(cfg):
    if cfg.MODEL.GENOTYPE is None:
        blocks_args = DEFAULT_BLOCKS_ARGS
    else:
        blocks_args = blocks_args_from_genotype(cfg.MODEL.GENOTYPE)

    d, w, drop_connect_rate = scaling_parameters(cfg.DATASET.INPUT_SHAPE)

    width_coefficient = cfg.MODEL.WIDTH_COEFFICIENT * w
    depth_coefficient = cfg.MODEL.DEPTH_COEFFICIENT * d
    depth_divisor = cfg.MODEL.DEPTH_DIVISOR
    head_filters = cfg.MODEL.HEAD_CHANNELS
    head_kernel = cfg.MODEL.HEAD_KERNEL
    head_activation = cfg.MODEL.HEAD_ACTIVATION
    keypoints = cfg.DATASET.OUTPUT_SHAPE[-1]
    regularizer = l2(cfg.TRAIN.WD)
    activation = cfg.MODEL.ACTIVATION

    img_input = layers.Input(shape=cfg.DATASET.INPUT_SHAPE)

    # Build stem
    x = img_input

    x = layers.Conv2D(
        round_filters(32, width_coefficient, depth_divisor), 3, 2,
        padding='same',
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        kernel_regularizer=regularizer,
        name='stem_conv')(x)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.Activation(activation, name='stem_activation')(x)

    # Build blocks
    blocks_args = copy.deepcopy(blocks_args)

    b = 0
    blocks = float(sum(round_repeats(args['repeats'], depth_coefficient) for args in blocks_args))
    for (i, args) in enumerate(blocks_args):
        assert args['repeats'] > 0
        # Update block input and output filters based on depth multiplier.
        args['filters_out'] = round_filters(args['filters_out'], width_coefficient, depth_divisor)
        repeats = args['repeats']
        args.pop('repeats')
        for j in range(round_repeats(repeats, depth_coefficient)):
            # The first block needs to take care of stride and filter size increase.
            if j > 0:
                args['strides'] = 1
            x = block(
                x,
                activation,
                drop_connect_rate * b / blocks,
                regularizer=regularizer,
                name='block{}{}_'.format(i + 1, chr(j + 97)),
                **args)
            b += 1

    for i in range(cfg.MODEL.HEAD_BLOCKS):
        x = layers.Conv2DTranspose(
            round_filters(head_filters, width_coefficient, depth_divisor),
            head_kernel,
            strides=2,
            padding='same',
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            kernel_regularizer=regularizer,
            name='head_block{}_conv'.format(i + 1))(x)
        x = layers.BatchNormalization(name='head_block{}_bn'.format(i + 1))(x)
        x = layers.Activation(head_activation, name='head_block{}_activation'.format(i + 1))(x)

    x = layers.Conv2D(
        keypoints,
        cfg.MODEL.FINAL_KERNEL,
        padding='same',
        use_bias=True,
        kernel_initializer=DENSE_KERNEL_INITIALIZER,
        kernel_regularizer=regularizer,
        name='final_conv')(x)

    model = Model(img_input, x, name='evopose')

    if cfg.MODEL.LOAD_WEIGHTS:
        if cfg.MODEL.PARENT is None:
            parent = get_parent(cfg.DATASET.INPUT_SHAPE)
        else:
            parent = load_model(cfg.MODEL.PARENT, compile=False)
        if parent is not None:
            model = transfer_params(parent, model)

    return model


def block(inputs,
          activation='swish',
          drop_rate=0.,
          name='',
          filters_out=16,
          kernel_size=3,
          strides=1,
          expand_ratio=1,
          se_ratio=0.,
          id_skip=True,
          project=True,
          regularizer=None):

    filters_in = inputs.shape[-1]

    # Expansion phase
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv2D(
            filters, 1,
            padding='same',
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            kernel_regularizer=regularizer,
            name=name + 'expand_conv')(inputs)
        x = layers.BatchNormalization(name=name + 'expand_bn')(x)
        x = layers.Activation(activation, name=name + 'expand_activation')(x)
    else:
        x = inputs

    # Depthwise Convolution
    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=strides,
        padding='same',
        use_bias=False,
        depthwise_initializer=CONV_KERNEL_INITIALIZER,
        kernel_regularizer=regularizer,
        name=name + 'dwconv')(x)
    x = layers.BatchNormalization(name=name + 'bn')(x)
    x = layers.Activation(activation, name=name + 'activation')(x)

    # Squeeze and Excitation phase
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = layers.GlobalAveragePooling2D(name=name + 'se_squeeze')(x)
        se = layers.Reshape((1, 1, filters), name=name + 'se_reshape')(se)
        se = layers.Conv2D(
            filters_se, 1,
            padding='same',
            activation=activation,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            kernel_regularizer=regularizer,
            name=name + 'se_reduce')(
            se)
        se = layers.Conv2D(
            filters, 1,
            padding='same',
            activation='sigmoid',
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            kernel_regularizer=regularizer,
            name=name + 'se_expand')(se)
        x = layers.multiply([x, se], name=name + 'se_excite')

    # Output phase
    if project:
        x = layers.Conv2D(
            filters_out, 1,
            padding='same',
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            kernel_regularizer=regularizer,
            name=name + 'project_conv')(x)
        x = layers.BatchNormalization(name=name + 'project_bn')(x)

    if id_skip and strides == 1 and filters_in == filters_out and project:
        if drop_rate > 0:
            x = layers.Dropout(
                drop_rate, noise_shape=(None, 1, 1, 1), name=name + 'drop')(x)
        x = layers.add([x, inputs], name=name + 'add')
    return x


def scaling_parameters(input_shape, default_size=224, alpha=1.2, beta=1.1, gamma=1.15):
    size = sum(input_shape[:2]) / 2
    if size <= 240:
        drop_connect_rate = 0.2
    elif size <= 300:
        drop_connect_rate = 0.3
    elif size <= 456:
        drop_connect_rate = 0.4
    else:
        drop_connect_rate = 0.5
    phi = (math.log(size) - math.log(default_size)) / math.log(gamma)
    d = alpha ** phi
    w = beta ** phi
    return d, w, drop_connect_rate


def array_in_list(arr, l):
    return next((True for elem in l if np.array_equal(elem, arr)), False)


def mutate(genotype, cache=[], max_stride=4):
    genotype = np.array(genotype)
    default_genotype = np.array(genotype_from_blocks_args(DEFAULT_BLOCKS_ARGS))
    mutant = genotype.copy()
    while np.array_equal(mutant, genotype) or array_in_list(mutant, cache):
        mutant = genotype.copy()
        block_id = np.random.randint(mutant.shape[0])
        gene_id = np.random.randint(mutant.shape[1])
        if gene_id == 0:  # kernel
            kernels = np.unique(default_genotype[:, 0])
            mutant[block_id][gene_id] = kernels[np.random.randint(len(kernels))]
        elif gene_id == 1:  # repeats
            if mutant[block_id][gene_id] == 1:
                mutant[block_id][gene_id] += 1
            elif mutant[block_id][gene_id] == np.max(default_genotype[:, 1]):
                mutant[block_id][gene_id] -= 1
            else:
                if np.random.uniform() < 0.5:
                    mutant[block_id][gene_id] += 1
                else:
                    mutant[block_id][gene_id] -= 1
        elif gene_id == 2:  # filters
            mutant[block_id][gene_id] = np.random.randint(1, default_genotype[block_id, 2] + 1)
        elif gene_id == 3:  # stride
            if block_id > 3:
                if mutant[block_id][gene_id] == 2 and np.sum(mutant[:, -1] - 1) == max_stride:
                    mutant[block_id][gene_id] = 1
                elif mutant[block_id][gene_id] == 1 and np.sum(mutant[:, -1] - 1) < max_stride:
                    mutant[block_id][gene_id] = 2
    print('block {}: {} -> {}'.format(block_id, genotype[block_id], mutant[block_id]))
    return list(mutant)


def layer_weights(model):
    names, weights = [], []
    for layer in model.layers:
        if layer.weights:
            names.append(layer.name)
            weights.append(layer.get_weights())
    return names, weights


def transfer_params(parent, child, disp=False):
    parent_layers, parent_weights = layer_weights(parent)
    for layer in child.layers:
        if layer.weights:
            if layer.name in parent_layers:
                parent_layer_weights = parent_weights[parent_layers.index(layer.name)]
                try:
                    # FULL WEIGHT TRANSFER
                    layer.set_weights(parent_layer_weights)
                except:
                    # PARTIAL WEIGHT TRANSFER
                    partial_weight_transfer(layer, parent_layer_weights, disp)

            else:
                # get last block with layer
                print('Repeat block:', layer.name)
                layer_name = '_'.join(layer.name.split('_')[1:])
                blocks_layer = [p for p in sorted(parent_layers)
                              if layer_name == '_'.join(p.split('_')[1:])
                              and 'block' in p]
                if len(blocks_layer) > 0:
                    parent_layer_weights = parent_weights[parent_layers.index(blocks_layer[-1])]
                    try:
                        # FULL WEIGHT TRANSFER
                        layer.set_weights(parent_layer_weights)
                    except:
                        # PARTIAL WEIGHT TRANSFER
                        partial_weight_transfer(layer, parent_layer_weights, disp)
                else:
                    if disp:
                        print('Did not transfer weights to {}'.format(layer.name))
    return child


if __name__ == '__main__':
    from utils import get_flops
    from dataset.coco import cn as cfg
    cfg.merge_from_file('../configs/evopose_s4_656_XS.yaml')
    model = EvoPose(cfg)
    print('{:.2f}M / {:.2f}G'.format(model.count_params() / 1e6, get_flops(model) / 1e9 / 2))

    # search space size:
    # default_genotype = np.array(genotype_from_blocks_args(DEFAULT_BLOCKS_ARGS))
    # s = 2**default_genotype.shape[0] * 4 ** default_genotype.shape[0]  # kernel and repeats
    # for c in default_genotype[:, 2]:  # channels
    #     s *= c
    # s *= 2**3  # stride
    # print('Search space size: 10^{:.0f}'.format(np.log10(s)))

    # np.random.seed(0)
    # cfg.MODEL.LOAD_WEIGHTS = False
    # parent_genotype = genotype_from_blocks_args(DEFAULT_BLOCKS_ARGS)
    # cfg.MODEL.GENOTYPE = parent_genotype
    # cache = [np.array(parent_genotype)]
    # parent = EvoPose(cfg)
    # print('{:.2f}M / {:.2f}G'.format(parent.count_params() / 1e6, get_flops(parent) / 1e9 / 2))
    # for i in range(200):
    #     np.random.seed(i + 200)
    #     child_genotype = mutate(cfg.MODEL.GENOTYPE, cache)
    #     cache.append(np.array(child_genotype))
    #     cfg.MODEL.GENOTYPE = child_genotype
    #     child = EvoPose(cfg)
    #     child = transfer_params(parent, child)
    #     print('{:.2f}M / {:.2f}G'.format(child.count_params() / 1e6, get_flops(child) / 1e9 / 2))
    #     print('-')
    #     parent = child
    #     del child
    #     K.clear_session()



