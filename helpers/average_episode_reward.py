from tf_agents.environments import TFPyEnvironment
from tf_agents.policies import tf_policy


def compute_avg_return_and_steps(environment: TFPyEnvironment, policy: tf_policy.TFPolicy, num_episodes=10):
    total_reward = 0.0
    total_steps = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_reward = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_reward += time_step.reward
            total_steps += 1
        total_reward += episode_reward

    avg_reward = total_reward / num_episodes
    avg_steps = total_steps / num_episodes
    return avg_reward.numpy()[0], avg_steps
