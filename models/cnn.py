from typing import Tuple
import tensorflow as tf
import numpy as np
from keras.engine.base_layer import Layer
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Reshape, Lambda, \
    Concatenate, ZeroPadding2D, AveragePooling2D
from keras.initializers.initializers_v2 import GlorotNormal, Zeros
from tensorflow.lite.python.schema_py_generated import Padding
from tf_agents.networks import sequential, nest_map, NestFlatten, Sequential

from models.residual_addition import add_residual


def create_cnn_model(model_params: dict, num_actions: int, observation_space: Tuple[int, ...]) -> Sequential:
    layers = []
    l2 = 'l2' if model_params['l2_regularisation'] else None

    layers.append(Lambda(lambda x: tf.cast(x, dtype=tf.float32), input_shape=observation_space))

    for i, m in enumerate(model_params['modules']):
        if m['type'] in ['pad-conv-pool', 'pad-conv']:
            layers.append(ZeroPadding2D(padding=(m['pad'], m['pad']), name=f'pad-{i + 1}'))

        layers.append(
            Conv2D(m['filter_count'], m['filter_size'],
                   kernel_initializer=GlorotNormal, kernel_regularizer=l2, name=f'conv-{i + 1}'))
        if model_params['batch_norm']:
            layers.append(BatchNormalization())
        layers.append(Activation(model_params['activation']))

        if m['type'] in ['conv-pool', 'pad-conv-pool']:
            if m['pool_func'] == 'max':
                layers.append(MaxPooling2D(m['pool_size'], name=f'avg-pool-{i + 1}'))
            elif m['pool_func'] == 'avg':
                layers.append(AveragePooling2D(m['pool_size'], name=f'max-pool-{i + 1}'))

        layers.append(Dropout(model_params['dropout']))

    layers.append(Flatten(name='flat-1'))

    if model_params['feed_input_after_cnn_layers']:
        layers = add_residual(layers,1)

    for i, neuron_count in enumerate(model_params['fc_layers']):
        layers.append(Dense(neuron_count, kernel_regularizer=l2, name=f'dense-{i}'))
        if model_params['batch_norm']:
            layers.append(BatchNormalization())
        layers.append(Activation(model_params['activation']))
        layers.append(Dropout(model_params['dropout']))

    if model_params['feed_input_to_last_layer']:
        layers = add_residual(layers,2)

    # Final layer has shape of number of actions we can make on the board.
    output_layer = Dense(num_actions, activation=None, kernel_initializer=GlorotNormal, bias_initializer=Zeros,
                         name='out', kernel_regularizer=l2)

    model = Sequential(layers + [output_layer])

    return model


# Does not implement stride
def get_parameter_shape(input_shape: Tuple[int, ...], module_config: dict) -> Tuple[int, ...]:
    if type(input_shape) is not tuple or len(input_shape) != 3:
        raise Exception(f'Expect 3D shape, but got {input_shape} instead.')

    if module_config['type'] == 'conv':
        # (N, H, W, C) -> (N, H', W', F)
        H, W, C = input_shape
        HH, WW = module_config['filter_size']
        H_s = 1 + H - HH
        W_s = 1 + W - WW
        F = module_config['filter_count']
        output_shape = (H_s, W_s, F)
        return output_shape

    if module_config['type'] == 'conv-pool':
        # (N, H, W, C) -> (N, H', W', F)
        conv_output = get_parameter_shape(input_shape, module_config | {'type': 'conv'})
        # (N, H', W', F) -> (N, H'', W'', F')
        H_s, W_s, F = conv_output
        H_ss = 1 + H_s - module_config['pool_size'][0]
        W_ss = 1 + W_s - module_config['pool_size'][1]
        output_shape = (H_ss, W_ss, F)
        return output_shape

    if module_config['type'] == 'pad-conv':
        H, W, C = input_shape
        H_p = module_config['pad'] * 2 + H
        W_p = module_config['pad'] * 2 + W
        padded_input = (H_p, W_p, C)
        return get_parameter_shape(padded_input, module_config | {'type': 'conv'})

    if module_config['type'] == 'pad-conv-pool':
        H, W, C = input_shape
        H_p = module_config['pad'] * 2 + H
        W_p = module_config['pad'] * 2 + W
        padded_input = (H_p, W_p, C)
        return get_parameter_shape(padded_input, module_config | {'type': 'conv-pool'})


def validate_model_parameters(model_params: dict, observation_space: Tuple[int, ...]):
    current_shape = observation_space
    for i, m in enumerate(model_params['modules']):
        output_shape = get_parameter_shape(current_shape, m)
        if np.any(np.array(output_shape) <= 1):
            raise Exception(f"Module {i} of type {m['type']} will yield an invalid output of {output_shape}"
                            f" given input {current_shape}")
        current_shape = output_shape


# m = {'activation': 'relu', 'batch_norm': True, 'dropout': 0.4, 'l2_regularisation': True,
#      'feed_input_to_last_layer': True, 'modules': [
#         {'type': 'conv-pool', 'filter_size': (1, 1), 'filter_count': 64, 'pool_func': 'max', 'pool_size': (2, 2)},
#         {'type': 'conv-pool', 'filter_size': (2, 2), 'filter_count': 128, 'pool_func': 'avg', 'pool_size': (2, 2)}],
#      'fc_layers': [128]}
#
# # validate_model_parameters(m, (5, 5, 4))
