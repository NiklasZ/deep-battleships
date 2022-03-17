from typing import Tuple
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.initializers.initializers_v2 import GlorotNormal, Zeros
from tf_agents.networks import sequential


# Tested 500 variations of an FC model and it won't even solve the episode most of the time:
# Top 50 model stats - mean: -98.47300720214844, min: -102.75, max: -86.9749984741211
def create_fc_model(model_params: dict, num_actions: int, observation_space: Tuple[int, ...]) -> Sequential:
    flattener = Flatten(input_shape=observation_space)

    layer_spec = model_params['layers']

    fc_layers = []
    for size in layer_spec:
        fc_layers.append(Dense(size, kernel_initializer=GlorotNormal, bias_initializer=Zeros))
        fc_layers.append(Activation(model_params['activation']))
        if model_params['batch_norm']:
            fc_layers.append(BatchNormalization())
        fc_layers.append(Dropout(model_params['dropout']))

    # Final layer has shape of number of actions we can make on the board.
    output_layer = Dense(num_actions, activation=None, kernel_initializer=GlorotNormal, bias_initializer=Zeros)

    model = sequential.Sequential([flattener] + fc_layers + [output_layer])

    return model
