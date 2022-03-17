from typing import Tuple
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Reshape, Lambda, \
    Concatenate
from keras.initializers.initializers_v2 import GlorotNormal, Zeros
from tf_agents.networks import sequential, nest_map, NestFlatten, Sequential


def create_res_cnn_model(model_params: dict, num_actions: int, observation_space: Tuple[int, ...]) -> Sequential:
    layers = []

    l2 = 'l2' if model_params['l2_regularisation'] else None

    # layers.append(Reshape(input_shape=observation_space, target_shape=(*observation_space, 1)))
    layers.append(
        Conv2D(32, (2, 2), input_shape=observation_space, kernel_initializer=GlorotNormal, name='conv1',
               kernel_regularizer=l2))
    layers.append(Activation(model_params['activation']))
    if model_params['batch_norm']:
        layers.append(BatchNormalization())
    layers.append(Dropout(model_params['dropout']))

    layers.append(MaxPooling2D((2, 2), name='max1'))

    layers.append(Conv2D(32, (1, 1), kernel_initializer=GlorotNormal, name='conv2', kernel_regularizer=l2))
    layers.append(Activation(model_params['activation']))
    if model_params['batch_norm']:
        layers.append(BatchNormalization())
    layers.append(Dropout(model_params['dropout']))

    layers.append(Flatten(name='flat1'))

    joint = []

    def duplicate_inputs(input):
        return {'conv_net': input, 'output_layer': input}

    joint.append(Lambda(duplicate_inputs)),
    joint.append(nest_map.NestMap({
        'conv_net': Sequential(layers),
        'output_layer': Flatten(),  # Lambda(lambda x: x)
    })),
    joint.append(NestFlatten())
    joint.append(Concatenate())
    # Final layer has shape of number of actions we can make on the board.
    output_layer = Dense(num_actions, activation=None, kernel_initializer=GlorotNormal, bias_initializer=Zeros,
                         name='out', kernel_regularizer=l2)

    model = sequential.Sequential(joint + [output_layer])

    return model
