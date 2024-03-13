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
DATASET_PATH = "dataset"
INDEX_FIELD = "timestamp"
DATA_FIELD = "num_request"
CPD_CANDIDATE_ROOT = "change_point_detection/offline_detection/cpd_candidate"


# %%
def get_data_file_list(dataset_path: str) -> List[str]:
    return os.listdir(dataset_path)

# %%


def read_dataset(csv_path: str, index_field: str, data_field: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    return df[index_field].to_numpy(), df[data_field].to_numpy()

# %%


def read_candidate_cpds(path: str) -> List[int]:
    candidate_cpds = None
    with open(path, "r") as f:
        candidate_cpds = json.load(f)
    return candidate_cpds


# %%
workload_to_skip_list = ["workload_1998-06-13", "workload_1998-06-14", "workload_1998-06-20", "workload_1998-06-21", "workload_1998-06-27", "workload_1998-06-28", "workload_1998-07-04"]

# %%


def get_flapping_time(n_pod_list: List, candidate_cpds: List):
    n_flipping = 0
    candidate_cpds.append(len(n_pod_list))
    seg_start = 0
    for seg_end in candidate_cpds:
        seg_idx = 0
        while seg_start+seg_idx+1 < seg_end:
            if n_pod_list[seg_start+seg_idx] != n_pod_list[seg_start+seg_idx+1]:
                n_flipping += 1
            seg_idx += 1
        seg_start = seg_end
    return n_flipping

# %%


def plot_flapping(workload: List, candidate_cpds: List, n_pod_list: List, title: str):
    fig, ax1 = plt.subplots()
    color_workload = "#3F51B5"  # material indigo
    color_cp = "#4CAF50"  # material green
    color_n_pod = "#F44336"  # material red
    x = np.arange(len(workload))
    line_workload, = ax1.plot(x, workload, color=color_workload, label="num_request_scaled_by_10x")
    for cp in candidate_cpds:
        ax1.axvline(x=cp, color=color_cp, linestyle='--', linewidth=1)
    ax1.set_xlabel('time (min)')
    ax1.set_ylabel('workload x (10 requests)')

    ax2 = ax1.twinx()
    line_n_pod, = ax2.plot(x, n_pod_list, color=color_n_pod, label="num_pod")

    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    lines = [line_workload, line_n_pod]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper right')

    if title is not None:
        ax1.set_title(title)

    return fig, ax1


# %%
data_file_list = get_data_file_list(DATASET_PATH)
results = ["ocpdas_simulation/reactive_result", "ocpdas_simulation/proactive_result", "ocpdas_simulation/ocpdas_result"]
for result in results:
    flapping_time_dict = {}
    workload_dict = None
    with open(result+"/"+"workload.json", "r") as f:
        workload_dict = json.load(f)
    n_pod_dict = None
    with open(result+"/"+"n_pod.json", "r") as f:
        n_pod_dict = json.load(f)
    for file_name in data_file_list:
        workload_name = file_name.split(".")[0]
        if workload_name in workload_to_skip_list:
            continue
        print("read %s" % (file_name))
        candidate_cpds = read_candidate_cpds(os.path.join(CPD_CANDIDATE_ROOT, workload_name+".json"))
        workload = workload_dict[workload_name]
        n_pod_list = n_pod_dict[workload_name]
        fig, ax = plot_flapping(workload, candidate_cpds, n_pod_list, workload)
        # fig.savefig(results+"/"+"npod_img/"+workload_name+".png")
        plt.show()
        flapping_time_dict[workload_name] = get_flapping_time(n_pod_list, candidate_cpds)
        with open(result+"/"+"flapping_time.json", "w") as f:
            json.dump(flapping_time_dict, f, indent=4)
