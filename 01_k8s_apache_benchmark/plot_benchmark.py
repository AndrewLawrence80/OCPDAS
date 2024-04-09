import matplotlib
import matplotlib.pyplot as plt
import json

font = {'size': 16}
matplotlib.rc('font', **font)

color_cpu = "#3f51b5"  # material indigo
color_fail = "#FF5722"  # material deep orange

rcpu_dict = None
with open("r-cpu.json", "r") as f:
    rcpu_dict = json.load(f)
failed_dict = None
with open("failure.json", "r") as f:
    failed_dict = json.load(f)

workload = [int(x) for x in list(rcpu_dict.keys())]
rcpu = [int(x) for x in list(rcpu_dict.values())]
failed = [int(x) for x in list(failed_dict.values())]

fig, ax1 = plt.subplots()
fig.set_size_inches(9,3)

ax1.plot(workload, rcpu, color=color_cpu)
ax1.scatter(workload, rcpu, color=color_cpu)
ax1.set_xlabel('request per minute')
ax1.set_ylabel('cpu utilization (m)')
line_cpu, = ax1.plot(workload, rcpu, color=color_cpu, label="cpu utilization")
ax1.grid(True, linestyle="--")

ax2 = ax1.twinx()
ax2.plot(workload, failed, color=color_fail)
ax2.scatter(workload, failed, color=color_fail)
ax2.set_ylabel('failed request number')
line_fail, = ax2.plot(workload, failed, color=color_fail, label="failed request number")

ax2.axvline(x=27500, color=color_fail, linestyle="--", linewidth=1)

ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

lines = [line_cpu, line_fail]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper left')
ax1.set_title("cpu utilizaiton - request benchmark")

fig.savefig("benchmark.pdf")

plt.show()
