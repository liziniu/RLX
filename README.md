
# RLX

## Overview

This is a reinforcement learning (RL) codebase based on TensorFlow without tedious ``sess.run()``. Its core components borrows **lunzi** module in [slbo](https://github.com/facebookresearch/slbo). With it, we can write TensorFlow in a Pythonic style. Overall, this codebase aims to provide a clean and easy tool to conduct research experiments on Atari and MuJoCo.

##  Algorithm

We don't aim to provide a universe interface for all algorithms on diverse environments. Thus, the listed algorithms are only implemented for specific tasks like Atari or MuJoCo.


- ACER[Atari]

- TRPO[MuJoCo]

- SAC[MuJoCo]

- GAIL[MuJoCo]

- TD3[MuJoCo]

- PPO[To be added]


These algorithms are tested on benchmark tasks, and the results can be found in the ``images`` folder.


## Usage

Example scripts are provided in the ``scripts`` folder.


## Reference

- baselines(https://github.com/openai/baselines)

- slbo(https://github.com/facebookresearch/slbo)

- dac(https://github.com/google-research/google-research/tree/master/dac)
