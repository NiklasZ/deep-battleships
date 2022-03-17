import gym
import gym_battleship
import numpy as np
from tf_agents.environments import suite_gym, tf_py_environment

# Environment Params for 10x10 scenario
from tf_agents.policies import random_tf_policy

from agents.train_agent import mask_invalid_actions
from helpers.step_metrics import calculate_step_metrics

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
    'get_invalid_action_mask': True
}


def play_games_with_random():
    py_env = suite_gym.load('Battleship-v0', gym_kwargs=E)
    tf_env = tf_py_environment.TFPyEnvironment(py_env)

    observation_and_action_constraint_splitter = mask_invalid_actions if E['get_invalid_action_mask'] else None
    random_policy = random_tf_policy.RandomTFPolicy(tf_env.time_step_spec(),
                                                    tf_env.action_spec(),
                                                    observation_and_action_constraint_splitter=observation_and_action_constraint_splitter)
    s_min, s_max, s_mean, s_median, _ = calculate_step_metrics(tf_env, random_policy, num_episodes=1_000)
    print(f'Metrics: min:{s_min}, max: {s_max}, mean: {s_mean}, median: {s_median}')


if __name__ == "__main__":
    play_games_with_random()
