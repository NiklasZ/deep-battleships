from collections import OrderedDict
import numpy as np
import tensorflow as tf
from agents.tune_agent import tune_agent

if len(tf.config.list_physical_devices('GPU')) == 0:
    raise Exception('No GPU found')

model_parameters = OrderedDict([
    ('type', 'CNN'),
    ('activation', ['elu', 'relu']),
    ('batch_norm', [True, False]),
    ('dropout', [0, 0.2, 0.4]),
    ('l2_regularisation', [True, False]),
    ('feed_input_to_last_layer', [True, False]),
    ('feed_input_after_cnn_layers', [True, False]),
    # TODO not implemented
    ('condense_sunk_matrix', False),  # Add the information of the sunk matrix to the missed matrix.

    # FC layer config
    ('n_fc_layers', [0, 1, 2]),
    ('neurons_per_layer', [64, 128, 256]),

    # Module config
    ('n_modules', [2, 3, 4]),
    ('module_combos', ['conv', 'conv-pool', 'pad-conv-pool', 'pad-conv']),
    ('filter_counts', [32, 64, 128, 256]),
    ('filter_sizes', [(1, 1), (2, 2), (3, 3), (4, 4)]),
    ('pool_funcs', ['max', 'avg']),
    ('pool_sizes', [(2, 2), (3, 3)]),
    ('pads', [0, 1, 2, 3]),
])

hyper_parameters = OrderedDict([
    # Constant
    ('num_iterations', 50_000),
    ('replay_buffer_max_length', 200_000),
    ('initial_collect_steps', 50_000),
    ('mask_invalid_actions', True),

    # Variable
    ('gamma', [0.99, 0.95, 0.9]),
    ('starting_epsilon', [0.1, 0.2, 0.5, 0.9, 1.0]),
    ('ending_epsilon', [0.05, 0.01, 0.001]),

    ('learning_rate', [1e-3, 1e-4, 1e-5]),
    ('learning_rate_decay', [1, 0.99, 0.98, 0.95]),
    ('decay_every', [10_000, 20_000, 50_000, 100_000]),
    ('batch_size', [16, 32, 64, 128]),
    ('target_update_tau', [1e-2, 1e-3, 1e-4]),

    ('model', model_parameters),
])

# Other Prams
P = {
    'collect_steps_per_iteration': 1,
    'log_interval': 200,
    'num_eval_episodes': 10,
    'eval_interval': 2000,
    'keep_top_k': 50,
    'tuning_runs': 2000,
    'run_alias': '10x10-cnn-tuning',
    'save_hyper_parameters': True,
}

# Environment Params for 5x5 scenario
# board_size = (5, 5)
# ship_sizes = {3: 1}

# # Environment Params for 7x7 scenario
# board_size = (7, 7)
# ship_sizes = {4: 1, 3: 1, 2: 1}
#
# # Environment Params for 10x10 scenario
board_size = (10, 10)
ship_sizes = {5: 1, 4: 1, 3: 2, 2: 1}

reward_dictionary = {
    'win': 100,  # for sinking all ships
    'missed': -1,  # for missing a shot
    'hit': 5,  # for hitting a ship
    'repeat_missed': -10,  # for shooting at an already missed cell
    'repeat_hit': -10  # for shooting at an already hit cell
}
E = {
    'ship_sizes': ship_sizes,
    'board_size': board_size,
    'episode_steps': np.prod(board_size),  # Number of steps until the episode terminates
    'reward_dictionary': reward_dictionary,
    'get_invalid_action_mask': hyper_parameters['mask_invalid_actions']
}

top_model_params = tune_agent(hyper_parameters=hyper_parameters, parameters=P, environment=E)
