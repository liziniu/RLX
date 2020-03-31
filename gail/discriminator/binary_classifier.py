import tensorflow as tf
from trpo.utils.normalizer import Normalizers
from acer.utils.cnn_utils import FCLayer
import lunzi.nn as nn
from typing import List


class BinaryClassifier(nn.Module):
    def __init__(self, dim_state: int, dim_action: int, hidden_sizes: List[int], normalizer: Normalizers,
                 save_normalizer=False):
        super().__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.hidden_sizes = hidden_sizes
        # this avoid to save normalizer into self.state_dict
        self.state_process_fn = lambda states_: normalizer.state(states_)
        self.action_process_fn = lambda actions_: actions_
        if save_normalizer:
            self.normalizer = normalizer

        with self.scope:
            self.op_states = tf.placeholder(tf.float32, [None, dim_state], "state")
            self.op_actions = tf.placeholder(tf.float32, [None, dim_action], "action")

            layers = []
            all_sizes = [dim_state + dim_action, *self.hidden_sizes]
            for i, (in_features, out_features) in enumerate(zip(all_sizes[:-1], all_sizes[1:])):
                layers.append(FCLayer(in_features, out_features))
                layers.append(nn.ReLU())
            layers.append(FCLayer(all_sizes[-1], 1))
            self.net = nn.Sequential(*layers)

            self.op_logits = self(self.op_states, self.op_actions)
            self.op_rewards = - tf.log(1-tf.nn.sigmoid(self.op_logits) + 1e-6)

    def forward(self, states: nn.Tensor, actions: nn.Tensor):
        inputs = tf.concat([
            self.state_process_fn(states), self.action_process_fn(actions)
        ], axis=-1)
        logits = self.net(inputs)[:, 0]
        return logits

    @nn.make_method(fetch='rewards')
    def get_reward(self, states, actions): pass

