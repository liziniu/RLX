import pickle
import os
import time
import tensorflow as tf
import numpy as np
import lunzi.nn as nn
from lunzi.Logger import logger, log_kvs
from trpo.policies.gaussian_mlp_policy import GaussianMLPPolicy
from trpo.v_function.mlp_v_function import MLPVFunction
from trpo.algos.trpo import TRPO
from trpo.utils.normalizer import Normalizers
from trpo.utils.runner import Runner, evaluate, gen_dtype
from gail.discriminator.discriminator import Discriminator
from gail.utils.mujoco_dataset import Mujoco_Dset
from utils import FLAGS, make_env, get_tf_config


"""Please Download Dataset from (https://drive.google.com/drive/folders/1h3H4AY_ZBx08hz-Ct0Nxxus-V1melu1U?usp=sharing).
"""


def main():
    FLAGS.set_seed()
    FLAGS.freeze()

    env = make_env(FLAGS.env.id, FLAGS.env.env_type, num_env=FLAGS.env.num_env, seed=FLAGS.seed, log_dir=FLAGS.log_dir,
                   rescale_action=FLAGS.env.rescale_action)
    env_eval = make_env(FLAGS.env.id, FLAGS.env.env_type, num_env=4, seed=FLAGS.seed+1000, log_dir=FLAGS.log_dir,
                        rescale_action=FLAGS.env.rescale_action)
    dim_state = env.observation_space.shape[0]
    dim_action = env.action_space.shape[0]

    normalizers = Normalizers(dim_action=dim_action, dim_state=dim_state)
    policy = GaussianMLPPolicy(dim_state, dim_action, FLAGS.TRPO.policy_hidden_sizes, normalizer=normalizers.state)
    vfn = MLPVFunction(dim_state, FLAGS.TRPO.vf_hidden_sizes, normalizers.state)
    algo = TRPO(vfn=vfn, policy=policy, dim_state=dim_state, dim_action=dim_action, **FLAGS.TRPO.algo.as_dict())

    discriminator = Discriminator(dim_state, dim_action, normalizers=normalizers, **FLAGS.GAIL.discriminator.as_dict())

    tf.get_default_session().run(tf.global_variables_initializer())

    # load expert dataset
    if not os.path.exists(FLAGS.GAIL.buf_load):
        raise FileNotFoundError('Expert dataset (%s) doest not exist' % FLAGS.GAIL.buf_load)
    expert_dataset = Mujoco_Dset(FLAGS.GAIL.buf_load, train_fraction=FLAGS.GAIL.train_frac,
                                 traj_limitation=FLAGS.GAIL.traj_limit)

    saver = nn.ModuleDict({'policy': policy, 'vfn': vfn, 'normalizers': normalizers})
    runner = Runner(env, max_steps=env.max_episode_steps, gamma=FLAGS.TRPO.gamma, lambda_=FLAGS.TRPO.lambda_)
    print(saver)

    max_ent_coef = FLAGS.TRPO.algo.ent_coef
    for t in range(0, FLAGS.GAIL.total_timesteps, FLAGS.TRPO.rollout_samples*FLAGS.GAIL.g_iters):
        time_st = time.time()
        if t % FLAGS.GAIL.eval_freq == 0:
            eval_returns, eval_lengths = evaluate(policy, env_eval)
            log_kvs(prefix='Evaluate', kvs=dict(
                iter=t, episode=dict(returns=np.mean(eval_returns), lengths=int(np.mean(eval_lengths)))
            ))

        # Generator
        generator_dataset = None
        for n_update in range(FLAGS.GAIL.g_iters):
            data, ep_infos = runner.run(policy, FLAGS.TRPO.rollout_samples)
            if FLAGS.TRPO.normalization:
                normalizers.state.update(data.state)
                normalizers.action.update(data.action)
                normalizers.diff.update(data.next_state - data.state)
            if t == 0 and n_update == 0:
                data_ = data.copy()
                data_ = data_.reshape([FLAGS.TRPO.rollout_samples//env.n_envs, env.n_envs])
                for e in range(env.n_envs):
                    samples = data_[:, e]
                    masks = 1 - (samples.done | samples.timeout)[..., np.newaxis]
                    masks = masks[:-1]
                    assert np.allclose(samples.state[1:] * masks, samples.next_state[:-1] * masks)
            t += FLAGS.TRPO.rollout_samples
            data.reward = discriminator.get_reward(data.state, data.action)
            advantages, values = runner.compute_advantage(vfn, data)
            train_info = algo.train(max_ent_coef, data, advantages, values)
            fps = int(FLAGS.TRPO.rollout_samples / (time.time() - time_st))
            train_info['reward'] = np.mean(data.reward)
            train_info['fps'] = fps
            log_kvs(prefix='TRPO', kvs=dict(
                iter=t, **train_info
            ))

            generator_dataset = data

        # Discriminator
        for n_update in range(FLAGS.GAIL.d_iters):
            batch_size = FLAGS.GAIL.d_batch_size
            d_train_infos = dict()
            for generator_subset in generator_dataset.iterator(batch_size):
                expert_state, expert_action = expert_dataset.get_next_batch(batch_size)
                train_info = discriminator.train(
                    expert_state, expert_action,
                    generator_subset.state, generator_subset.action
                )
                for k, v in train_info.items():
                    if k not in d_train_infos:
                        d_train_infos[k] = []
                    d_train_infos[k].append(v)
            d_train_infos = {k: np.mean(v) for k, v in d_train_infos.items()}
            if n_update == FLAGS.GAIL.d_iters - 1:
                log_kvs(prefix='Discriminator', kvs=dict(
                    iter=t, **d_train_infos
                ))

        if t % FLAGS.TRPO.save_freq == 0:
            np.save('{}/stage-{}'.format(FLAGS.log_dir, t), saver.state_dict())
            np.save('{}/final'.format(FLAGS.log_dir), saver.state_dict())
    np.save('{}/final'.format(FLAGS.log_dir), saver.state_dict())


if __name__ == '__main__':
    with tf.Session(config=get_tf_config()):
        main()