import os
import functools
import gym
from lunzi.Logger import logger
from .monitor import Monitor
from .atari_wrapper import make_atari, wrap_deepmind
from .batched_env import SubprocVecEnv, DummyVecEnv
from .mujoco_wrapper import ReScaleActionWrapper

__all__ = ['make_env']


COUNT = 0


def make_env(env_id: str, env_type: str, num_env: int, seed: int, log_dir: str, **kwargs):
    if env_type == 'atari':
        make_thunk = make_atari_env
    elif env_type == 'mujoco':
        if kwargs.get('rescale_action', None):
            logger.info('MuJoCo Rescale action...')
        make_thunk = functools.partial(make_mujoco_env, rescale_action=kwargs.get('rescale_action', False))
    else:
        make_thunk = make_gym_env
    if num_env == 1:
        env = DummyVecEnv([functools.partial(make_thunk, env_id=env_id, seed=seed, index=index, log_dir=log_dir)
                           for index in range(num_env)])
    else:
        env = SubprocVecEnv([functools.partial(make_thunk, env_id=env_id, seed=seed, index=index, log_dir=log_dir)
                             for index in range(num_env)])
    global COUNT
    COUNT += num_env
    return env


def make_gym_env(env_id: str, seed: int, index: int, log_dir: str, allow_early_resets=True):
    env = gym.make(env_id)
    global COUNT
    seed = COUNT * 10 + index * 100 + 1000 + seed
    env.seed(seed)
    try:
        env.action_space.seed(seed)
    except AttributeError:
        pass
    env = Monitor(env, os.path.join(log_dir, str(index+COUNT)), allow_early_resets=allow_early_resets)
    return env


def make_atari_env(env_id: str, seed: int, index: int, log_dir: str, allow_early_resets=True):
    env = make_atari(env_id)
    global COUNT
    seed = COUNT * 10 + index * 100 + 1000 + seed
    env.seed(seed)
    try:
        env.action_space.seed(seed)
    except AttributeError:
        pass
    env = Monitor(env, os.path.join(log_dir, str(index+COUNT)), allow_early_resets=allow_early_resets)
    env = wrap_deepmind(env, frame_stack=True)
    return env


def make_mujoco_env(env_id: str, seed: int, index: int, log_dir: str, allow_early_resets=True, rescale_action=True):
    env = gym.make(env_id)
    global COUNT
    seed = COUNT * 10 + index * 100 + 1000 + seed
    env.seed(seed)
    try:
        env.action_space.seed(seed)
    except AttributeError:
        pass
    env = Monitor(env, os.path.join(log_dir, str(index+COUNT)), allow_early_resets=allow_early_resets)
    if rescale_action:
        env = ReScaleActionWrapper(env)
    return env
