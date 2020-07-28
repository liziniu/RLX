import tensorflow as tf
import numpy as np
import lunzi.nn as nn
from trpo.utils.normalizer import Normalizers
from .binary_classifier import BinaryClassifier
from typing import List


class Discriminator(nn.Module):
    def __init__(self, dim_state: int, dim_action: int, hidden_sizes: List[int], normalizers: Normalizers,
                 lr: float, ent_coef: float, max_grad_norm: None, subsampling_rate=20.):
        super().__init__()
        self.ent_coef = ent_coef
        self.subsampling_rate = subsampling_rate

        with self.scope:
            self.op_true_states = tf.placeholder(tf.float32, [None, dim_state], "true_state")
            self.op_true_actions = tf.placeholder(tf.float32, [None, dim_action], "true_action")
            self.op_fake_states = tf.placeholder(tf.float32, [None, dim_state], "fake_state")
            self.op_fake_actions = tf.placeholder(tf.float32, [None, dim_action], "fake_actions")
            self.op_true_masks = tf.placeholder(tf.float32, [None], "mask")

            self.classifier = BinaryClassifier(dim_state, dim_action, hidden_sizes, normalizers, save_normalizer=False)

            self.op_loss, self.op_classifier_loss, self.op_entropy_loss, self.op_true_prob, self.op_fake_prob = \
                self(self.op_true_states, self.op_true_actions, self.op_fake_states, self.op_fake_actions,
                     self.op_true_masks)

            optimizer = tf.train.AdamOptimizer(lr)
            params = self.classifier.parameters()
            grads_and_vars = optimizer.compute_gradients(self.op_loss, var_list=params)
            if max_grad_norm is not None:
                clip_grads, op_grad_norm = tf.clip_by_global_norm([grad for grad, _ in grads_and_vars], max_grad_norm)
                clip_grads_and_vars = [(grad, var) for grad, (_, var) in zip(clip_grads, grads_and_vars)]
            else:
                op_grad_norm = tf.global_norm([grad for grad, _ in grads_and_vars])
                clip_grads_and_vars = grads_and_vars
            self.op_train = optimizer.apply_gradients(clip_grads_and_vars)
            self.op_grad_norm = op_grad_norm

    def forward(self, true_states: nn.Tensor, true_actions: nn.Tensor, fake_states: nn.Tensor, fake_actions: nn.Tensor,
                true_masks: nn.Tensor):
        true_logits = self.classifier(true_states, true_actions)
        fake_logits = self.classifier(fake_states, fake_actions)

        true_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=true_logits, labels=tf.ones_like(true_logits)
        )
        fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_logits, labels=tf.zeros_like(fake_logits)
        )
        true_masks = tf.maximum(0., -true_masks)
        true_weight = true_masks / self.subsampling_rate + (1 - true_masks)

        classify_loss = tf.reduce_mean(true_loss * true_weight) + tf.reduce_mean(fake_loss)
        logits = tf.concat([true_logits, fake_logits], axis=0)
        entropy = (1. - tf.nn.sigmoid(logits)) * logits + tf.nn.softplus(-logits)
        entropy_loss = -self.ent_coef * tf.reduce_mean(entropy)

        loss = classify_loss + entropy_loss
        true_prob = tf.nn.sigmoid(true_logits)
        fake_prob = tf.nn.sigmoid(fake_logits)
        return loss, classify_loss, entropy_loss, true_prob, fake_prob

    @nn.make_method(fetch='loss')
    def get_loss(self, true_states, true_actions, fake_states, fake_actions, true_masks): pass

    def get_reward(self, states, actions):
        return self.classifier.get_reward(states, actions)

    def train(self, true_states, true_actions, fake_states, fake_actions, true_masks=None):
        if true_masks is None:
            true_masks = np.zeros([len(true_states), ], dtype=np.float32)
        _, loss, classifier_loss, entropy_loss, true_prob, fake_prob, grad_norm = self.get_loss(
            true_states, true_actions, fake_states, fake_actions, true_masks,
            fetch='train loss classifier_loss entropy_loss true_prob fake_prob grad_norm'
        )
        info = dict(
            loss=np.mean(loss),
            classifier_loss=np.mean(classifier_loss),
            entropy_loss=np.mean(entropy_loss),
            grad_norm=np.mean(grad_norm),
            true_prob=np.mean(true_prob),
            fake_prob=np.mean(fake_prob),
        )
        return info

