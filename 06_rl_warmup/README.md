# 06 RL Warmup

The directory `06_rl_wamrup` contains code and results related to running RL to gain prior knowledge when changes are about to happen on the WorldCup dataset. It consists of the following files:

1. `model.py`: Basic Q Network using LSTM.
2. `replay_buffer.py`: Replay buffer to store past experience in Double DQN training.
3. `agent.py`: Double DQN agent.
4. `world_cup_env.py`: The training environment derived from the `Env` class in the package `gymnasium`.
5. `train_utils.py`: Off-policy training proxy.
6. `run_rl.ipynb`: Code to train the Double DQN agent.
7. `eval_rl.ipynb`: Code to evaluate the training performance of the Double DQN.
8. `state_dict`: Save directory of the Q Network state dict after training.
9. `saved_reward`: Save directory of the training reward of the Double DQN agent.

## How to use

1. Run `run_rl.ipynb`.
2. Run `eval_rl.ipynb`.
