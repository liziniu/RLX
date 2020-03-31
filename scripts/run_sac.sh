
set -e
set -x

ENV=Hopper-v2
NUM_ENV=1


if [ "$(uname)" == "Darwin" ]; then
  python -m sac.main -s \
    algorithm=sac \
    env.id=${ENV} \
    env.num_env=${NUM_ENV} \
    env.env_type=mujoco
elif [ "$(uname)" == "Linux" ]; then
  for ENV in Hopper-v2 HalfCheetah-v2 Ant-v2
  do
  for SEED in 100 200 300
    do
      python -m sac.main -s \
      algorithm=sac \
      seed=${SEED} \
      env.id=${ENV} \
      env.num_env=${NUM_ENV} \
      env.env_type=mujoco & sleep 2
    done
    wait
  done
fi
