import time
import collections
import tensorflow as tf
import numpy as np
import lunzi.nn as nn
from lunzi.dataset import Dataset
from lunzi.Logger import logger, log_kvs
from utils import FLAGS, make_env, get_tf_config
from trpo.utils.runner import gen_dtype, evaluate
from td3.policies.critic import Critic
from td3.policies.actor import Actor
from td3.algos.td3 import TD3


def main():
    FLAGS.set_seed()
    FLAGS.freeze()

    env = make_env(FLAGS.env.id, FLAGS.env.env_type, num_env=FLAGS.env.num_env, seed=FLAGS.seed, log_dir=FLAGS.log_dir,
                   rescale_action=FLAGS.env.rescale_action)
    env_eval = make_env(FLAGS.env.id, FLAGS.env.env_type, num_env=4, seed=FLAGS.seed+1000, log_dir=FLAGS.log_dir)
    dim_state = env.observation_space.shape[0]
    dim_action = env.action_space.shape[0]

    actor = Actor(dim_state, dim_action, hidden_sizes=FLAGS.TD3.actor_hidden_sizes)
    critic = Critic(dim_state, dim_action, hidden_sizes=FLAGS.TD3.critic_hidden_sizes)
    td3 = TD3(dim_state, dim_action, actor=actor, critic=critic, **FLAGS.TD3.algo.as_dict())

    tf.get_default_session().run(tf.global_variables_initializer())
    td3.update_actor_target(tau=0.0)
    td3.update_critic_target(tau=0.0)

    dtype = gen_dtype(env, 'state action next_state reward done timeout')
    buffer = Dataset(dtype=dtype, max_size=FLAGS.TD3.buffer_size)
    saver = nn.ModuleDict({'actor': actor, 'critic': critic})
    print(saver)

    n_steps = np.zeros(env.n_envs)
    n_returns = np.zeros(env.n_envs)

    train_returns = collections.deque(maxlen=40)
    train_lengths = collections.deque(maxlen=40)
    states = env.reset()
    time_st = time.time()
    for t in range(FLAGS.TD3.total_timesteps):
        if t < FLAGS.TD3.init_random_steps:
            actions = np.array([env.action_space.sample() for _ in range(env.n_envs)])
        else:
            raw_actions = actor.get_actions(states)
            noises = np.random.normal(loc=0., scale=FLAGS.TD3.explore_noise, size=raw_actions.shape)
            actions = np.clip(raw_actions + noises, -1, 1)
        next_states, rewards, dones, infos = env.step(actions)
        n_returns += rewards
        n_steps += 1
        timeouts = n_steps == env.max_episode_steps
        terminals = np.copy(dones)
        for e, info in enumerate(infos):
            if info.get('TimeLimit.truncated', False):
                terminals[e] = False

        transitions = [states, actions, next_states.copy(), rewards, terminals, timeouts.copy()]
        buffer.extend(np.rec.fromarrays(transitions, dtype=dtype))

        indices = np.where(dones | timeouts)[0]
        if len(indices) > 0:
            next_states[indices] = env.partial_reset(indices)

            train_returns.extend(n_returns[indices])
            train_lengths.extend(n_steps[indices])
            n_returns[indices] = 0
            n_steps[indices] = 0
        states = next_states.copy()

        if t == 2000:
            assert env.n_envs == 1
            samples = buffer.sample(size=None, indices=np.arange(2000))
            masks = 1 - (samples.done | samples.timeout)[..., np.newaxis]
            masks = masks[:-1]
            assert np.allclose(samples.state[1:] * masks, samples.next_state[:-1] * masks)

        if t >= FLAGS.TD3.init_random_steps:
            samples = buffer.sample(FLAGS.TD3.batch_size)
            train_info = td3.train(samples)
            if t % FLAGS.TD3.log_freq == 0:
                fps = int(t / (time.time() - time_st))
                train_info['fps'] = fps
                log_kvs(prefix='TD3', kvs=dict(
                    iter=t, episode=dict(
                        returns=np.mean(train_returns) if len(train_returns) > 0 else 0.,
                        lengths=int(np.mean(train_lengths) if len(train_lengths) > 0 else 0)),
                    **train_info))

        if t % FLAGS.TD3.eval_freq == 0:
            eval_returns, eval_lengths = evaluate(actor, env_eval, deterministic=False)
            log_kvs(prefix='Evaluate', kvs=dict(
                iter=t, episode=dict(returns=np.mean(eval_returns), lengths=int(np.mean(eval_lengths)))
            ))

        if t % FLAGS.TD3.save_freq == 0:
            np.save('{}/stage-{}'.format(FLAGS.log_dir, t), saver.state_dict())
            np.save('{}/final'.format(FLAGS.log_dir), saver.state_dict())

    np.save('{}/final'.format(FLAGS.log_dir), saver.state_dict())


if __name__ == '__main__':
    with tf.Session(config=get_tf_config()):
        main()
