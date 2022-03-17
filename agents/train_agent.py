import os
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
import tf_agents as tfa
import tensorflow_probability as tfp
from tensorflow.python.ops.summary_ops_v2 import create_file_writer
from tf_agents.agents import tf_agent, DqnAgent
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.networks import Sequential, mask_splitter_network
from tf_agents.policies import random_tf_policy, py_tf_eager_policy, PolicySaver
from tf_agents.specs import BoundedTensorSpec
from tf_agents.typing.types import Tensor, Splitter
from tf_agents.utils import common

from helpers.average_episode_reward import compute_avg_return_and_steps
from helpers.replay_buffer import start_replay_server, cleanup_replay_server
import gym
import gym_battleship
import h5py

from models.masking_net import MaskedQNetwork


def mask_invalid_actions(observation: Tensor) -> Splitter:
    # 1st matrix has 1 for explored cells and 0 for unexplored.
    # As there is no point to make actions on already explored cells,
    # We can use the matrix as a mask.
    # action_mask = (np.flatten(observation[..., 0]) - 1) * -1
    # action_mask = tf.compat.v1.layers.flatten(tf.equal(observation[..., 0], 0.0))
    return observation['observation'], observation['valid_actions']


def train_agent(id: int, q_net: Sequential, hyper_parameters: dict, environment_parameters: dict,
                other_parameters: dict, policy_dir: Optional[str] = None, log_dir: Optional[str] = None) -> \
        Tuple[DqnAgent, List[float], List[float]]:
    file_writer = None
    if log_dir:
        run_dir = log_dir + f'/{id}-metrics'
        os.mkdir(run_dir)
        file_writer = create_file_writer(run_dir)
        file_writer.set_as_default()

    H = hyper_parameters
    P = other_parameters
    b1, b2 = environment_parameters['board_size']
    best_steps = (b1 * b2 * (3/4))  # corresponds to a 3/4 of an episode.

    env = suite_gym.load('Battleship-v0', gym_kwargs=environment_parameters)
    train_py_env = suite_gym.load('Battleship-v0', gym_kwargs=environment_parameters)
    eval_py_env = suite_gym.load('Battleship-v0', gym_kwargs=environment_parameters)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)  # Convert the env to be entirely in tensorflow
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    global_step = tf.compat.v1.train.get_or_create_global_step()

    # NOTE: as the framework is very opaque, we can use this to debug calls
    # to the neural net.
    # if hyper_parameters['mask_invalid_actions']:
    #     # q_net = mask_splitter_network.MaskSplitterNetwork(
    #     #     splitter_fn=mask_invalid_actions,
    #     #     wrapped_network=q_net,
    #     #     passthrough_mask=False,
    #     # )
    #     q_net = MaskedQNetwork(env.observation_spec(), q_net)
    #     target_q_net = MaskedQNetwork(env.observation_spec(), target_q_net)

    # Learning rate
    learning_rate = tf.compat.v1.train.exponential_decay(
        learning_rate=tf.constant(H['learning_rate'], dtype=tf.float32),
        global_step=global_step,
        decay_steps=tf.constant(H['decay_every'], dtype=tf.float32),
        decay_rate=tf.constant(H['learning_rate_decay'], dtype=tf.float32),
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Epsilon control
    epsilon = tf.compat.v1.train.polynomial_decay(
        learning_rate=tf.constant(H['starting_epsilon'], dtype=tf.float32),
        global_step=global_step,
        decay_steps=H['num_iterations'],
        end_learning_rate=tf.constant(H['ending_epsilon'], dtype=tf.float32),
    )

    train_step_counter = tf.Variable(0, dtype=tf.int64)

    observation_and_action_constraint_splitter = mask_invalid_actions if H['mask_invalid_actions'] else None

    agent = dqn_agent.DdqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        gamma=H['gamma'],
        # target_q_network=target_q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter,
        epsilon_greedy=epsilon,
        # target_update_period=H['target_update_period'],
        target_update_tau=H['target_update_tau'],
        observation_and_action_constraint_splitter=observation_and_action_constraint_splitter,
        summarize_grads_and_vars=False  # Enable if you want a lot of logs and a fraction of the training speed.
    )

    agent.initialize()

    policy_writer = None
    if policy_dir:
        policy_writer = PolicySaver(agent.policy, batch_size=None)

    # Prepare a replay buffer
    rb_observer, replay_buffer, replay_server = start_replay_server(agent=agent, replay_buffer_max_length=H[
        'replay_buffer_max_length'])

    # Populate the replay buffer with some results from a random policy
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec(),
                                                    observation_and_action_constraint_splitter=observation_and_action_constraint_splitter)
    print(f"Collecting {H['initial_collect_steps']} environment steps randomly...")
    py_driver.PyDriver(
        env,
        py_tf_eager_policy.PyTFEagerPolicy(
            random_policy, use_tf_function=True, batch_time_steps=False),
        # agent.collect_policy, use_tf_function=True),
        [rb_observer],
        max_steps=H['initial_collect_steps'],
    ).run(train_py_env.reset())

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=H['batch_size'],
        num_steps=2).prefetch(3)
    iterator = iter(dataset)

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step.
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return, avg_steps = compute_avg_return_and_steps(eval_env, agent.policy, num_episodes=P['num_eval_episodes'])
    returns = [avg_return]
    steps = [avg_steps]

    # Reset the environment.
    time_step = train_py_env.reset()

    # Create a driver to collect experience.
    collect_driver = py_driver.PyDriver(
        env,
        py_tf_eager_policy.PyTFEagerPolicy(
            agent.collect_policy, use_tf_function=True),
        [rb_observer],
        max_steps=P['collect_steps_per_iteration'])

    for i in range(H['num_iterations']):
        # Collect a few steps and save to the replay buffer.
        time_step, _ = collect_driver.run(time_step)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % P['log_interval'] == 0:
            print(f'step = {step}, loss = {train_loss}')

        if step % P['eval_interval'] == 0:
            avg_reward, avg_steps = compute_avg_return_and_steps(eval_env, agent.policy, P['num_eval_episodes'])
            print(f'step = {step}, Average Steps = {avg_steps}, Avg Reward = {avg_reward}')
            returns.append(avg_reward)
            steps.append(avg_steps)
            # Write metrics to tensorboard log
            if log_dir:
                tf.summary.scalar('average steps', data=avg_steps, step=tf.cast(i, dtype=tf.int64))
                tf.summary.scalar('average reward', data=avg_reward, step=tf.cast(i, dtype=tf.int64))

            if policy_dir and avg_steps < best_steps:
                print(f'Saved model for new highest steps of {avg_steps}')
                policy_writer.save(f'{policy_dir}/{id}')
                best_steps = avg_steps

    cleanup_replay_server(observer=rb_observer, server=replay_server)

    if log_dir:
        file_writer.close()

    return agent, returns, steps
