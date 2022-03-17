import os
from collections import OrderedDict
from datetime import datetime
from typing import List

import numpy as np
from tf_agents.environments import suite_gym

from agents.train_agent import train_agent
from helpers.random_hyper_picker import pick_random_hyper_parameters
from models.cnn import create_cnn_model, validate_model_parameters


def tune_agent(hyper_parameters: OrderedDict, parameters: dict, environment: dict) -> List[dict]:
    top_k_models = []
    already_chosen_hypers = set()
    debugging = False # Set to true when debugging to errors are thrown correctly.
    P, E = parameters, environment

    date_string = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    output_folder = f"data/1_tuning/{date_string}-{P['run_alias']}"
    log_folder = f'{output_folder}/logs'
    policy_folder = f'{output_folder}/policies'

    for p in [output_folder, log_folder, policy_folder]:
        if not os.path.exists(p):
            os.mkdir(p)

    env = suite_gym.load('Battleship-v0', gym_kwargs=E)

    for t in range(1, P['tuning_runs'] + 1):
        print(f"\n--- Tuning Run {t}/{P['tuning_runs']} ---\n")
        H = pick_random_hyper_parameters(hyper_parameters, already_chosen_hypers)
        M = H['model']
        print(f'Picked hyper-parameters:')
        print(H)

        # Sometimes tf will fail with NaN values, interrupting tuning.
        # And on other occasions, the generated CNN may not be compatible with the input size.
        try:
            observation_shape = env.observation_space.spaces['observation'].shape if hyper_parameters['mask_invalid_actions'] else env.observation_space.shape
            validate_model_parameters(model_params=M, observation_space=observation_shape)
            q_net = create_cnn_model(model_params=M, num_actions=env.action_space.n,
                                     observation_space=observation_shape)

            agent, rewards, steps = train_agent(id=t, q_net=q_net, hyper_parameters=H, environment_parameters=E,
                                                other_parameters=P,
                                                policy_dir=policy_folder, log_dir=log_folder)
            last_5_avg_reward = np.mean(rewards[-5])
            best_reward = np.max(rewards)
            reward_variance = np.var(rewards)
            last_5_avg_steps = np.mean(steps[-5])
            best_steps = np.max(steps)

            top_k_models.append(
                {'id': t,
                 'best_reward': best_reward, 'last_5_reward_avg': last_5_avg_reward, 'reward_variance': reward_variance,
                 'best_steps': best_steps, 'last_5_avg_steps': last_5_avg_steps,
                 'hyper_parameters': H, 'net': q_net, 'agent': agent})

            if len(top_k_models) > P['keep_top_k']:
                top_k_models = sorted(top_k_models, key=lambda x: x['best_steps'])[:-1]
                steps = [x['best_steps'] for x in top_k_models]
                print(
                    f"\nTop {P['keep_top_k']} model best step stats - mean: {np.mean(steps)}, min: {np.min(steps)}, max: {np.max(steps)}")

            if P['save_hyper_parameters']:
                # Save everything except for the net to a text file
                top_k_models = sorted(top_k_models, key=lambda x: x['best_steps'])
                top_params = [str(dict(filter(lambda x: x[0] not in ['net', 'agent'], d.items()))) for d in
                              top_k_models]
                file_path = f"{output_folder}/top_parameters.json"
                with open(file_path, 'w') as f:
                    f.write('\n'.join(top_params))
                    print(f"Saving top {len(top_k_models)} model hyper parameters to '{file_path}'")

        except Exception as e:
            if not debugging:
                print(e)
            else:
                raise e

    print('\nTop Model:')
    print(f"best steps: {top_k_models[0]['best_steps']}, last 5 avg: {top_k_models[0]['last_5_avg_steps']}")
    print('Hyper_parameters: ')
    print(top_k_models[0]['hyper_parameters'])
    print('Structure')
    top_k_models[0]['net'].summary()
    return top_k_models
