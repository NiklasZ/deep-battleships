import tensorflow as tf

# class MaskingLayer(tf.keras.layers.Layer):
#     def __init__(self, num_outputs):
#         super(MyDenseLayer, self).__init__()
#         self.num_outputs = num_outputs
#
#     def build(self, input_shape):
#         self.kernel = self.add_weight("kernel",
#                                       shape=[int(input_shape[-1]),
#                                              self.num_outputs])
#
#     def call(self, inputs):
#         return tf.matmul(inputs, self.kernel)
from tf_agents.networks import network
from tf_agents.networks.q_network import QNetwork


class MaskedQNetwork(network.Network):
    def __init__(self,
                 observation_spec,
                 q_net: QNetwork,
                 mask_q_value=-(10 ** 5),
                 name='MaskQNetwork'):
        super(MaskedQNetwork, self).__init__(input_tensor_spec=observation_spec,
                                             state_spec=(),
                                             name=name)

        self._q_net = q_net
        self._mask_q_value = mask_q_value

    def call(self, observation, step_type=None, network_state=None):
        state = observation['observation']
        mask = observation['valid_actions']
        q_values, _ = self._q_net(state, step_type)

        # Sometimes tf calls this without any data, so we have to avoid making calculations then.
        if step_type is not None and step_type.shape[0] is not None:
            small_constant = tf.constant(self._mask_q_value, dtype=q_values.dtype,
                                         shape=q_values.shape)
            zeros = tf.zeros(shape=mask.shape, dtype=mask.dtype)
            masked_q_values = tf.where(tf.math.equal(zeros, mask),
                                       small_constant, q_values)
        else:
            masked_q_values = q_values

        return masked_q_values, network_state
