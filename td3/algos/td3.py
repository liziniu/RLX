import lunzi.nn as nn
from lunzi.nn import Tensor
import tensorflow as tf
import numpy as np
from td3.policies.actor import Actor
from td3.policies.critic import Critic


class TD3(nn.Module):
    def __init__(self, dim_state, dim_action, actor: Actor, critic: Critic, gamma: float,
                 actor_lr: float, critic_lr: float, tau: float,
                 policy_noise: float, policy_noise_clip: float, policy_update_freq: int):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.policy_noise = policy_noise
        self.policy_noise_clip = policy_noise_clip
        self.policy_update_freq = policy_update_freq

        with self.scope:
            self.op_states = tf.placeholder(tf.float32, [None, dim_state], 'states')
            self.op_actions = tf.placeholder(tf.float32, [None, dim_action], 'actions')
            self.op_next_states = tf.placeholder(tf.float32, [None, dim_state], 'next_states')
            self.op_rewards = tf.placeholder(tf.float32, [None], 'rewards')
            self.op_terminals = tf.placeholder(tf.float32, [None], 'terminals')
            self.op_tau = tf.placeholder(tf.float32, [], 'tau')

            actor_target = actor.clone()
            target_params, source_params = actor_target.parameters(), actor.parameters()
            self.op_update_actor_target = tf.group(
                *[tf.assign(v_t, self.op_tau * v_t + (1 - self.op_tau) * v_s)
                  for v_t, v_s in zip(target_params, source_params)])
            self.actor_target = actor_target

            critic_target = critic.clone()
            target_params, source_params = critic_target.parameters(), critic.parameters()
            self.op_update_critic_target = tf.group(
                *[tf.assign(v_t, self.op_tau * v_t + (1 - self.op_tau) * v_s)
                  for v_t, v_s in zip(target_params, source_params)])
            self.critic_target = critic_target

            self.op_actor_loss, self.op_critic_loss = self(
                self.op_states, self.op_actions, self.op_next_states, self.op_rewards, self.op_terminals
            )

            actor_optimizer = tf.train.AdamOptimizer(learning_rate=self.actor_lr)
            actor_grads = tf.gradients(self.op_actor_loss, self.actor.parameters())
            actor_grads_and_vars = list(zip(actor_grads, self.actor.parameters()))
            self.op_actor_train = actor_optimizer.apply_gradients(actor_grads_and_vars)
            self.op_actor_grad_norm = tf.global_norm(actor_grads_and_vars)

            critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_lr)
            critic_grads = tf.gradients(self.op_critic_loss, self.critic.parameters())
            critic_grads_and_vars = list(zip(critic_grads, critic.parameters()))
            self.op_critic_train = critic_optimizer.apply_gradients(critic_grads_and_vars)
            self.op_critic_grad_norm = tf.global_norm(critic_grads_and_vars)

            # self.op_actor_train = actor_optimizer.minimize(self.op_actor_loss, var_list=self.actor.parameters())
            # self.op_critic_train = critic_optimizer.minimize(self.op_critic_loss, var_list=self.critic.parameters())

        self.iterations = 0
        self.last_actor_loss = 0
        self.last_actor_grad_norm = 0

    def forward(self, states: Tensor, actions: Tensor, next_states: Tensor, rewards: Tensor, terminals: Tensor):
        # actor
        q1, _ = self.critic(states, self.actor(states))
        actor_loss = -tf.reduce_mean(q1)

        # critic
        target_action_noise = tf.random.normal(tf.shape(actions), stddev=self.policy_noise)
        target_action_noise = tf.clip_by_value(target_action_noise, -self.policy_noise_clip, self.policy_noise_clip)
        noisy_action_targets = self.actor_target(next_states) + target_action_noise

        clipped_noisy_action_targets = tf.clip_by_value(noisy_action_targets, -1, 1)
        q_next1, q_next2 = self.critic_target(next_states, clipped_noisy_action_targets)
        q_next = tf.reduce_min(tf.concat([q_next1[:, None], q_next2[:, None]], axis=-1), axis=-1)
        q_target = tf.stop_gradient(rewards + self.gamma * (1 - terminals) * q_next)
        q_pred1, q_pred2 = self.critic(states, actions)
        critic_loss = tf.reduce_mean(tf.square(q_pred1 - q_target)) + tf.reduce_mean(tf.square(q_pred2 - q_target))

        return actor_loss, critic_loss

    @nn.make_method(fetch='update_critic_target')
    def update_critic_target(self, tau): pass

    @nn.make_method(fetch='update_actor_target')
    def update_actor_target(self, tau): pass

    @nn.make_method(fetch='critic_train critic_loss')
    def optimize_critic(self, states, actions, next_states, rewards, terminals): pass

    @nn.make_method(fetch='actor_train actor_loss')
    def optimize_actor(self, states): pass

    def train(self, data):
        _, critic_loss, critic_grad_norm = self.optimize_critic(
            data.state, data.action, data.next_state, data.reward, data.done,
            fetch='critic_train critic_loss critic_grad_norm'
        )

        self.iterations += 1

        if self.iterations % self.policy_update_freq == 0:
            _, actor_loss, actor_grad_norm = self.optimize_actor(
                data.state,
                fetch='actor_train actor_loss actor_grad_norm'
            )
            self.last_actor_loss = actor_loss
            self.last_actor_grad_norm = actor_grad_norm

            self.update_actor_target(tau=self.tau)
            self.update_critic_target(tau=self.tau)
        else:
            actor_loss = self.last_actor_loss
            actor_grad_norm = self.last_actor_grad_norm

        info = dict(
            dsc_mean=np.mean(data.done),
            actor_loss=actor_loss,
            actor_grad_norm=actor_grad_norm,
            critic_loss=critic_loss,
            critic_grad_norm=critic_grad_norm,
        )
        return info

