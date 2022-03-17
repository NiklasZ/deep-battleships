from collections import OrderedDict
from typing import Set, Union, List

import numpy as np


def pick_randomly(candidates: Union[float, int, List]):
    if isinstance(candidates, list):
        idx = np.random.choice(len(candidates))
        return candidates[idx]
    else:
        return candidates


def pick_random_hyper_parameters(hyper_parameters: OrderedDict, already_chosen_hypers: Set[str]) -> dict:
    chosen_parameters = OrderedDict()
    for key, value in hyper_parameters.items():
        if key == 'model':
            if hyper_parameters[key]['type'] == 'MLP':
                chosen_parameters[key] = pick_random_mlp_parameters(value)
            else:
                chosen_parameters[key] = pick_random_cnn_parameters(value)
        else:
            chosen_parameters[key] = pick_randomly(value)

    combined = str(chosen_parameters)

    if combined in already_chosen_hypers:
        return pick_random_hyper_parameters(hyper_parameters, already_chosen_hypers)

    already_chosen_hypers.add(combined)
    return chosen_parameters


def pick_random_mlp_parameters(model_parameters: dict) -> dict:
    chosen_parameters = {}
    if model_parameters['type'] != 'MLP':
        raise Exception(f"Cannot generate model parameters for unknown type {model_parameters['type']}")

    regular_parameters = dict(
        filter(lambda x: x[0] not in ['type', 'n_layers', 'neurons_per_layer'], model_parameters.items()))

    layer_count = pick_randomly(model_parameters['n_layers'])
    layers = []
    for _ in range(layer_count):
        neuron_count = pick_randomly(model_parameters['neurons_per_layer'])
        layers.append(neuron_count)

    chosen_parameters['layers'] = layers

    for key, value in regular_parameters.items():
        chosen_parameters[key] = pick_randomly(value)

    return chosen_parameters

def pick_random_cnn_parameters(model_parameters: dict) -> dict:
    chosen_parameters = {}
    if model_parameters['type'] != 'CNN':
        raise Exception(f"Cannot generate model parameters for unknown type {model_parameters['type']}")

    regular_parameters = dict(
        filter(lambda x: x[0] not in ['type', 'n_fc_layers', 'neurons_per_layer', 'n_modules', 'module_combos',
                                      'filter_counts', 'filter_sizes', 'pool_funcs', 'pool_sizes', 'pads'],
               model_parameters.items()))

    for key, value in regular_parameters.items():
        chosen_parameters[key] = pick_randomly(value)

    module_count = pick_randomly(model_parameters['n_modules'])
    modules = []

    for _ in range(module_count):
        m = {
            'type': pick_randomly(model_parameters['module_combos']),
            'filter_size': pick_randomly(model_parameters['filter_sizes']),
            'filter_count': pick_randomly(model_parameters['filter_counts'])
        }
        if m['type'] in ['conv-pool', 'pad-conv-pool']:
            m['pool_func'] = pick_randomly(model_parameters['pool_funcs'])
            m['pool_size'] = pick_randomly(model_parameters['pool_sizes'])

        if m['type'] in ['pad-conv-pool', 'pad-conv']:
            m['pad'] = pick_randomly(model_parameters['pads'])

        modules.append(m)

    chosen_parameters['modules'] = modules

    layer_count = pick_randomly(model_parameters['n_fc_layers'])
    layers = []
    for _ in range(layer_count):
        neuron_count = pick_randomly( model_parameters['neurons_per_layer'])
        layers.append(neuron_count)

    chosen_parameters['fc_layers'] = layers

    return chosen_parameters

