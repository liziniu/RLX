
# This is an script to run gail using the dataset provided by OpenAI Baselines.
set -e
set -x

ENV=Hopper
NUM_ENV=1
ROLLOUT_SAMPLES=1000
BUF_LOAD=~/project/deterministic.trpo.${ENV}.0.00.npz
VF_HIDDEN_SIZES=100
POLICY_HIDDEN_SIZES=100
TRPO_ENT_COEF=0.0


if [ "$(uname)" == "Darwin" ]; then
  python -m gail.main -s \
    algorithm=gail \
    env.id=${ENV}-v2 \
    env.num_env=${NUM_ENV} \
    env.env_type=mujoco \
    GAIL.buf_load=${BUF_LOAD} \
    TRPO.rollout_samples=${ROLLOUT_SAMPLES} \
    TRPO.vf_hidden_sizes=${VF_HIDDEN_SIZES} \
    TRPO.policy_hidden_sizes=${POLICY_HIDDEN_SIZES} \
    TRPO.algo.ent_coef=${TRPO_ENT_COEF}
elif [ "$(uname)" == "Linux" ]; then
  for ENV in  Hopper HalfCheetah Walker2d
  do
    BUF_LOAD=~/project/deterministic.trpo.${ENV}.0.00.npz
    for SEED in 100 200 300
    do
      python -m gail.main -s \
        seed=${SEED} \
        algorithm=gail \
        env.id=${ENV}-v2 \
        env.num_env=${NUM_ENV} \
        env.env_type=mujoco \
        GAIL.buf_load=${BUF_LOAD} \
        TRPO.rollout_samples=${ROLLOUT_SAMPLES} \
        TRPO.vf_hidden_sizes=${VF_HIDDEN_SIZES} \
        TRPO.policy_hidden_sizes=${POLICY_HIDDEN_SIZES} \
        TRPO.algo.ent_coef=${TRPO_ENT_COEF} & sleep 2
    done
    wait
  done
fi
