from typing import Tuple
from reverb import Server
from tf_agents.agents import tf_agent
from tf_agents.replay_buffers import reverb_replay_buffer, ReverbReplayBuffer, ReverbAddTrajectoryObserver
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
import reverb


def start_replay_server(agent: tf_agent.TFAgent, replay_buffer_max_length: int) -> Tuple[
    ReverbAddTrajectoryObserver, ReverbReplayBuffer, Server]:
    table_name = 'uniform_table'
    replay_buffer_signature = tensor_spec.from_spec(
        agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(
        replay_buffer_signature)

    table = reverb.Table(
        table_name,
        max_size=replay_buffer_max_length,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature)

    reverb_server = reverb.Server([table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        agent.collect_data_spec,
        table_name=table_name,
        sequence_length=2,
        local_server=reverb_server)

    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
        replay_buffer.py_client,
        table_name,
        sequence_length=2)

    return rb_observer, replay_buffer, reverb_server


def cleanup_replay_server(observer: ReverbAddTrajectoryObserver, server: Server):
    observer.close()
    server.stop()
