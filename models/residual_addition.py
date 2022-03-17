from typing import List
from keras import Sequential
from keras.layers import Lambda, Flatten, Concatenate
from keras.type.types import Layer
from tf_agents.networks import nest_map, NestFlatten
import tensorflow as tf


def add_residual(layers: List[Layer], id=1) -> List[Layer]:
    joint = []

    def duplicate_inputs(input):
        return {'previous_net': input, 'output_layer': input}

    joint.append(Lambda(duplicate_inputs, name=f'lambda-{id}')),
    joint.append(nest_map.NestMap({
        'previous_net': Sequential(layers, name=f'seq-{id}'),
        'output_layer': Sequential([Flatten(name=f'flat-{id+1}'), Lambda(lambda x: tf.cast(x, dtype=tf.float32))])
    })),
    joint.append(NestFlatten())
    joint.append(Concatenate(name=f'cat-{id}'))

    return joint
