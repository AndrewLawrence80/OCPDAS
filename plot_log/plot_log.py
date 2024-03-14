import os
import matplotlib
import matplotlib.pyplot as plt
from scanf import scanf

font = {'size': 16}

matplotlib.rc('font', **font)


def plot_log(workloads, cpu_usage, title=None):
    fig, ax1 = plt.subplots()

    fig.set_size_inches(16, 9)

    x = range(len(workloads))

    color = "#3f51b5"  # material indigo
    ax1.set_xlabel('time (min)')
    ax1.set_ylabel('request number x (10 requests)')
    line_workload, = ax1.plot(x, workloads, color=color, label="request number scaled by 10x")
    ax1.grid(True,linestyle="--")

    ax2 = ax1.twinx()

    color = '#F44336'  # material red
    ax2.set_ylabel("cpu utilization (m)")
    line_cpu, = ax2.plot(x, cpu_usage, color=color, label="cpu utilization")

    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    lines = [line_workload, line_cpu]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper right')

    if title is not None:
        ax1.set_title(title)

    return fig


data_path = "log"

for log in os.listdir(data_path):
    date_str = log.split(".")[0]
    with open(os.path.join(data_path, log), "r") as f:
        lines = f.readlines()
        workloads = []
        cpu_usage = []
        for idx in range(len(lines)):
            if idx % 2 != 0:
                continue
            workload, cpu = scanf("workload %d, cpu utilization %dm", lines[idx])
            workloads.append(workload)
            cpu_usage.append(cpu)
        fig = plot_log(workloads, cpu_usage, title="workload "+date_str)
        fig.savefig("plot_log/img/"+date_str+".pdf")
        plt.close()
