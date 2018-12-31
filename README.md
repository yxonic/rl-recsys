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
fret config env DeepSPEnv -dataset zhixue
fret config sp_model EERNN
fret config         # check current configuration
fret train_env      # calls SPEnv.train

fret config ValueBasedTrainer
fret config agent DQN
fret config policy SimpleNet
fret train_agent    # calls Trainer.train
```

You can also copy and/or edit `config.toml` directly to setup modules.

## Visualization

You can use TensorBoard to visualize training loss, just by typing:

```sh
tensorboard --logdir ws 
```

## TODOs
- [x] Load datasets: questions, words, knowledge, records
- [x] `DeepSPEnv.train`: train deep score prediction models on records
- [ ] Sample students
- [x] Command `train_agent`
- [ ] Policy Gradient and Actor Critic methods
- [ ] Reward function
