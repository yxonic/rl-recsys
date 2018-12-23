# rl-recsys
RL Recommendation System

## Install
```
pip install -r requirements.txt
```

## Usage

```sh
mkdir -p ws/dqn-eernn-exp1
cd ws/dqn-eernn-exp1
fret config env DeepSPEnv
fret config sp_model EERNN
fret config         # check current configuration
fret train_env      # calls SPEnv.train

fret config Trainer
fret config agent DQN
fret train_agent    # calls Trainer.train
```

You can also copy and/or edit `config.toml` directly to setup modules.
