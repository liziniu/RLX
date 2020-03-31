
set -e
set -x

ENV=BreakoutNoFrameskip-v4
NUM_ENV=4


if [ "$(uname)" == "Darwin" ]; then
  python -m acer.main -s \
    algorithm=acer\
    env.id=${ENV} \
    env.num_env=${NUM_ENV} \
    env.env_type=atari
elif [ "$(uname)" == "Linux" ]; then
  for ENV in BreakoutNoFrameskip-v4 PongNoFrameskip-v4 KungFuMasterNoFrameskip-v4
  do
    for SEED in 100 200 300
    do
      python -m acer.main -s \
      algorithm=acer \
      seed=${SEED} \
      env.id=${ENV} \
      env.num_env=${NUM_ENV} \
      env.env_type=atari  & sleep 2
    done
    wait
  done
fi
