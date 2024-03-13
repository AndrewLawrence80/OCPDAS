# %%
import os
import pandas as pd
from typing import List, Tuple
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
import torch

font = {'size': 16}

matplotlib.rc('font', **font)

# %%
DATASET_PATH = "dataset"
INDEX_FIELD = "timestamp"
DATA_FIELD = "num_request"
CPD_CANDIDATE_ROOT = "change_point_detection/offline_detection/cpd_candidate"


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
class Simulator:
    def __init__(self, workload: np.ndarray, capacity_per_pod: float, sla: float, fluctuation: float, t_cooldown: int = 1) -> None:
        self.t = 0
        self.sla = sla
        self.upper_sla = sla+fluctuation
        self.lower_sla = sla-fluctuation
        self.workload = workload
        self.capaciy_per_pod = capacity_per_pod
        self.t_cooldown = t_cooldown
        self.r_cpu_list = []
        self.n_pod_list = []

    def run(self):
        n_pod = int(np.ceil(self.workload[0]/(self.sla*self.capaciy_per_pod)))
        t_last_scale = -1
        while self.t < len(self.workload):
            current_workload = self.workload[self.t]
            r_cpu = 1.0*current_workload/(n_pod*self.capaciy_per_pod)
            self.n_pod_list.append(n_pod)
            self.r_cpu_list.append(r_cpu)
            if self.t-t_last_scale > self.t_cooldown:
                target_n_pod = int(np.ceil(n_pod*(r_cpu/self.sla)))
                if target_n_pod != n_pod:
                    n_pod = target_n_pod
                    t_last_scale = self.t
            self.t += 1

# %%
data_file_list = get_data_file_list(DATASET_PATH)
x, y = [], []
for file_name in data_file_list:
    workload_name = file_name.split(".")[0]
    print("read %s" % (file_name))
    np_index, np_data = read_dataset(os.path.join(DATASET_PATH, file_name), INDEX_FIELD, DATA_FIELD)
    np_data = np_data/10.0
    simulator = Simulator(workload=np_data, capacity_per_pod=2500, sla=0.5, fluctuation=0.1, t_cooldown=1)
    simulator.run()
    break


