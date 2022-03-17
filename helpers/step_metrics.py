from typing import Tuple, List

import numpy as np
from tf_agents.environments import TFPyEnvironment
from tf_agents.policies import tf_policy
from tqdm import tqdm


def calculate_step_metrics(environment: TFPyEnvironment, policy: tf_policy.TFPolicy, num_episodes=10) -> Tuple[
    float, float, float, float, List[int]]:
    step_history = []

    for _ in tqdm(range(num_episodes)):

        time_step = environment.reset()
        steps = 0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            steps += 1
        step_history.append(steps)

    min_steps = np.min(step_history)
    max_steps = np.max(step_history)
    mean_steps = np.mean(step_history)
    median_steps = np.median(step_history)

    return min_steps, max_steps, mean_steps, median_steps, step_history
