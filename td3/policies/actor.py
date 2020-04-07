from typing import List
import lunzi.nn as nn
import tensorflow as tf
from acer.utils.cnn_utils import FCLayer


class Actor(nn.Module):
    def __init__(self, dim_state, dim_action, hidden_sizes: List[int]):
        super().__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.hidden_sizes = hidden_sizes

        with self.scope:
            self.op_states = tf.placeholder(tf.float32, shape=[None, dim_state], name='states')

            layers = []
            all_sizes = [dim_state, *self.hidden_sizes]
            for i, (in_features, out_features) in enumerate(zip(all_sizes[:-1], all_sizes[1:])):
                layers.append(FCLayer(in_features, out_features))
                layers.append(nn.ReLU())
            layers.append(FCLayer(all_sizes[-1], dim_action, init_scale=0.01))
            layers.append(nn.Tanh())
            self.net = nn.Sequential(*layers)
            self.op_actions = self(self.op_states)

    def forward(self, states: nn.Tensor):
        actions = self.net(states)
        return actions

    @nn.make_method(fetch='actions')
    def get_actions(self, states): pass

    def clone(self):
        return Actor(self.dim_state, self.dim_action, self.hidden_sizes)
