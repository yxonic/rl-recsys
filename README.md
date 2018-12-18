# rl-recsys
RL Recommendation System

## Install
```
pip install -r requirements.txt
```

## Usage

```sh
mkdir -p ws/dqn-ekt-exp1
cd ws/dqn-ekt-exp1
fret config Trainer <args...>
fret config agent DQN <args...>
fret config env EKT <args...>
fret train_env
fret train_agent
```
