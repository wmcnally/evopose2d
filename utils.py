import os
import os.path as osp
import pickle
import sys
import tempfile
from contextlib import contextmanager

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def detect_hardware(tpu_name):
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)  # TPU detection
    except ValueError:
        tpu = None
        gpus = tf.config.experimental.list_logical_devices("GPU")

    # Select appropriate distribution strategy
    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    elif len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
        print('Running on multiple GPUs ', [gpu.name for gpu in gpus])
    elif len(gpus) == 1:
        strategy = tf.distribute.get_strategy()  # default strategy that works on CPU and single GPU
        print('Running on single GPU ', gpus[0].name)
    else:
        strategy = tf.distribute.get_strategy()  # default strategy that works on CPU and single GPU
        print('Running on CPU')
    print("Number of accelerators: ", strategy.num_replicas_in_sync)
    return tpu, strategy


def get_flops(model, write_path=tempfile.NamedTemporaryFile().name):
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        if write_path:
            opts['output'] = 'file:outfile={}'.format(write_path)  # suppress output
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops


def add_regularization(model, regularizer=tf.keras.regularizers.l2(0.0001)):
    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
        print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
        return model

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = tf.keras.models.model_from_json(model_json)

    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model


def partial_weight_transfer(child_layer, parent_weights, disp):
    child_weights = child_layer.get_weights()
    for i, child_weight in enumerate(child_weights):
        parent_weight = parent_weights[i]

        if disp:
            print('Transferring partial weights for layer {}: {} -> {}'.format(
                child_layer.name, parent_weight.shape, child_weight.shape))

        # CONVOLUTION
        if len(child_weight.shape) == 4:
            # (child kernel size, input channels child, output channels child), ...
            (kc, icc, occ), (kp, icp, ocp) = child_weight.shape[1:], parent_weight.shape[1:]

            if (icc > icp and occ > ocp) or (icc > icp and occ == ocp) or (
                    icc == icp and occ > ocp):
                if kc == kp:
                    child_weights[i][:, :, :icp, :ocp] = parent_weight
                elif kc < kp:
                    p = (kp - kc) // 2  # pad
                    child_weights[i][:, :, :icp, :ocp] = parent_weight[p:p + kc, p:p + kc, :, :]
                elif kc > kp:
                    p = (kc - kp) // 2
                    child_weights[i][p:p + kp, p:p + kp, :icp, :ocp] = parent_weight

            elif (icc < icp and occ > ocp) or (icc < icp and occ == ocp):
                if kc == kp:
                    child_weights[i][:, :, :, :ocp] = parent_weight[:, :, :icc, :]
                elif kc < kp:
                    p = (kp - kc) // 2  # pad
                    child_weights[i][:, :, :, :ocp] = parent_weight[p:p + kc, p:p + kc, :icc, :]
                elif kc > kp:
                    p = (kc - kp) // 2
                    child_weights[i][p:p + kp, p:p + kp, :, :ocp] = parent_weight[:, :, :icc, :]

            elif (icc > icp and occ < ocp) or (icc == icp and occ < ocp):
                if kc == kp:
                    child_weights[i][:, :, :icp, :] = parent_weight[:, :, :, :occ]
                elif kc < kp:
                    p = (kp - kc) // 2  # pad
                    child_weights[i][:, :, :icp, :] = parent_weight[p:p + kc, p:p + kc, :, :occ]
                elif kc > kp:
                    p = (kc - kp) // 2
                    child_weights[i][p:p + kp, p:p + kp, :icp, :] = parent_weight[:, :, :, :occ]

            elif icc < icp and occ < ocp:
                if kc == kp:
                    child_weights[i] = parent_weight[:, :, :icc, :occ]
                elif kc < kp:
                    p = (kp - kc) // 2  # pad
                    child_weights[i] = parent_weight[p:p + kc, p:p + kc, :icc, :occ]
                elif kc > kp:
                    p = (kc - kp) // 2
                    child_weights[i][p:p + kp, p:p + kp, :, :] = parent_weight[:, :, :icc, :occ]

        # DENSE
        elif len(child_weight.shape) == 2:
            icc, icp = child_weight.shape[0], parent_weight.shape[0]
            if icc < icp:
                child_weights[i] = parent_weight[:icc, :]
            else:
                weight_filler = np.zeros((icc - icp, child_weight.shape[1]))
                child_weights[i] = np.concatenate((parent_weight, weight_filler), axis=0)

        # BATCH NORM
        elif len(child_weight.shape) == 1:
            icc, icp = child_weight.shape[0], parent_weight.shape[0]
            if icc < icp:
                child_weights[i] = parent_weight[:icc]
            else:
                weight_filler = np.zeros((icc - icp,))
                weight_filler[:] = np.mean(parent_weight)
                child_weights[i] = np.concatenate((parent_weight, weight_filler), axis=0)
    try:
        child_layer.set_weights(child_weights)
    except:

        print("Partial weight transfer failed for '{}'".format(child_layer.name))


def get_models(run_dir):
    meta_files, saved_models, genotypes = [], [], {}
    gens = sorted(os.listdir(run_dir))
    gens = [g for g in gens if '.pkl' not in g]
    for g in gens:
        gen_models = sorted(os.listdir(osp.join(run_dir, g)))
        gen_meta = [m for m in gen_models if 'meta' in m]
        gen_models = [m for m in gen_models if '.h5' in m]
        meta_files.extend(gen_meta)
        saved_models.extend(gen_models)
        for m in gen_meta:
            meta_data = pickle.load(open(osp.join(run_dir, g, m), 'rb'))
            genotypes[np.int(m.split('_')[1])] = meta_data['config'].MODEL.GENOTYPE
    return meta_files, saved_models, genotypes
