""" defines utility functions. """

import sys
import json
import numpy as np
import pickle as pkl
import logging
import tensorflow as tf


# json loads strings as unicode; we currently still work with Python 2 strings, and need conversion
def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key, value) in d.items())


def load_dict(filename):
    try:
        with open(filename, 'r') as f:
            # return unicode_to_utf8(json.load(f))
            return json.load(f)
    except:
        with open(filename, 'r') as f:
            return pkl.load(f)


def load_config(basename):
    try:
        with open('{:s}.json'.format(basename), 'rb') as f:
            return json.load(f)
    except:
        try:
            with open('{:s}.pkl'.format(basename), 'rb') as f:
                return pkl.load(f)
        except:
            sys.stderr.write('Error: config file {:s}.json is missing\n'.format(basename))
            sys.exit(1)


def seq2words(seq, inverse_target_dictionary, join=True):
    seq = np.array(seq, dtype='int64')
    assert len(seq.shape) == 1
    return factoredseq2words(seq.reshape([seq.shape[0], 1]),
                             [inverse_target_dictionary],
                             join)


def factoredseq2words(seq, inverse_dictionaries, join=True):
    assert len(seq.shape) == 2
    assert len(inverse_dictionaries) == seq.shape[1]
    words = []
    eos_reached = False
    for i, word in enumerate(seq):
        if eos_reached:
            break
        factors = []
        for j, factor in enumerate(word):
            if factor == 0:
                # TODO: DEBUGGING
                # factors.append(inverse_dictionaries[j][factor])
                eos_reached = True
                break
            elif factor in inverse_dictionaries[j]:
                factors.append(inverse_dictionaries[j][factor])
            else:
                factors.append('UNK')
        word = '|'.join(factors)
        words.append(word)
    return ' '.join(words) if join else words


def reverse_dict(dictt):
    keys, values = zip(*dictt.items())
    r_dictt = dict(zip(values, keys))
    return r_dictt


# Parallel training helpers

# see https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
def get_visible_gpus():
    """ Returns a list of the identifiers of all visible GPUs. """
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [device.name for device in local_device_protos if device.device_type == 'GPU']


# see https://github.com/tensorflow/tensorflow/issues/9517
def assign_to_device(controller_device, operator_device):
    """ Returns a function to place variables on the best-suited device for multi-GPU execution.
    controller_device: Device to put the variables on. Example values are /GPU:0 and /CPU:0.
    operator_device: Device for everything but variables. """
    variable_ops = ['Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable', 'MutableHashTableOfTensors',
                    'MutableDenseHashTable']

    def _assign_op(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in variable_ops:
            return controller_device
        else:
            return operator_device

    return _assign_op


def get_devices(num_gpus):
    """ Returns devices available for training. """
    # CPU is assigned the role of the controller device
    controller = '/cpu:0'
    # Single-device case
    operators = list()
    if num_gpus <= 1:
        operators.append('/device:GPU:0' if num_gpus == 1 else '/cpu:0')
    # Multi-GPU case
    else:
        visible_gpus = get_visible_gpus()
        operators = ['/device:GPU:{:d}'.format(gpu_id) for gpu_id in range(num_gpus)]

        if num_gpus > len(visible_gpus):
            logging.info('WARNING: {:d} GPUs have been designated for use by the model, '
                         'while only {:d} GPUs are available! Using {:d} GPUs.'
                         .format(num_gpus, len(visible_gpus), len(visible_gpus)))
    return controller, operators


# https://stackoverflow.com/questions/38160940/how-to-count-total-number-of-trainable-parameters-in-a-tensorflow-model
def count_parameters(config, float_dtype):
    """ Estimates the number of trainable parameters in the loaded model; note - activation sizes are not computed!"""
    mbyte_per_batch = None
    gbyte_per_batch = None
    estimate_memory = True
    # Consider different float sizes
    if float_dtype == tf.bfloat16 or float_dtype == tf.float16:
        float_multiply = 2
    elif float_dtype == tf.float32:
        float_multiply = 4
    elif float_dtype == tf.float64:
        float_multiply = 8
    else:
        float_multiply = 1
        estimate_memory = False
    # Calculate the number of model parameters + activations and, optionally, the associated memory footprint
    total_parameters = int(np.sum([np.prod(var.shape) for var in tf.trainable_variables()]))

    # TODO: Add a means for the calculation of activations
    model_size = total_parameters
    if estimate_memory and config.token_batch_size <= 0:
        parameter_memory = (total_parameters * float_multiply) / (1024 ** 2) * 3  # assuming two copies for Adam
        batch_size = \
            (config.token_batch_size / config.max_len) if config.token_batch_size > 0 else config.sentence_batch_size
        mbyte_per_batch = parameter_memory * batch_size
        gbyte_per_batch = mbyte_per_batch / 1024
    return model_size, mbyte_per_batch, gbyte_per_batch
