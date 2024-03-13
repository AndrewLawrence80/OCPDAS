# %%
import os
import pandas as pd
from typing import List, Tuple
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
from torch.utils.data import DataLoader
from model import LSTMQNet
import torch
from train_utils import OffPolicyTrainer
import torch.nn as nn
from world_cup_env import WorldCupEnv
from agent import DoubleDQNAgent

font = {'size': 16}

matplotlib.rc('font', **font)

# %%
DATASET_PATH = "dataset"
INDEX_FIELD = "timestamp"
DATA_FIELD = "num_request"
CPD_CANDIDATE_ROOT = "change_point_detection/offline_detection/cpd_candidate"
N_LOOKBACK = 4
N_PREDICT = 2
BATCH_SIZE = 16

LR = 1e-2
N_EPOCHS = 200
# SCHEDULER_MILESTONE=[150,180]
# SCHEDULER_GAMMA=0.3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# %%
def get_data_file_list(dataset_path: str) -> List[str]:
    return os.listdir(dataset_path)

# %%
def read_dataset(csv_path: str,index_field:str,data_field:str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    return df[index_field].to_numpy(), df[data_field].to_numpy()

# %%
def read_candidate_cpds(path: str) -> List[int]:
    candidate_cpds = None
    with open(path, "r") as f:
        candidate_cpds = json.load(f)
    return candidate_cpds

# %%
data_file_list = get_data_file_list(DATASET_PATH)
x, y = [], []
for file_name in data_file_list:
    workload_name = file_name.split(".")[0]
    print("read %s" % (file_name))
    np_index, np_data = read_dataset(os.path.join(DATASET_PATH, file_name), INDEX_FIELD, DATA_FIELD)
    np_data = np_data/20000.0
    np_data = np.diff(np_data)
    np_data = np_data.reshape((-1, 1))
    candidate_cpds = read_candidate_cpds(os.path.join(CPD_CANDIDATE_ROOT, workload_name+".json"))
    env = WorldCupEnv(np_data, candidate_cpds, N_LOOKBACK, N_PREDICT)
    agent = DoubleDQNAgent(3)
    trainer = OffPolicyTrainer(env, agent, num_episodes=100, replay_buffer_size=128, batch_size=32, discount_factor=0.9, epsilon_start=0.5, epsilon_end=0.1, epsilon_step=20, learning_rate_start=1e-3, learning_rate_end=1e-4, learning_rate_step=100, tau=0.05)
    trainer.train()
    break


