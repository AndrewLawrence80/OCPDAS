import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
start_time = []
end_time = []
# with open("world_cup_time.txt", "r") as f:
#     for line in f.readlines():
#         line = str.strip(line)
#         time_obj = datetime.strptime(line, "%Y-%m-%d %H:%M:%S")
#         time_obj = time_obj-timedelta(hours=2)
#         start_time.append((time_obj-timedelta(minutes=20)).strftime("%Y-%m-%d %H:%M:%S"))
#         end_time.append((time_obj+timedelta(minutes=140)).strftime("%Y-%m-%d %H:%M:%S"))
with open("segment_start_time.txt","r") as f:
    for line in f.readlines():
        line=str.strip(line)
        time_obj = datetime.strptime(line, "%Y-%m-%d %H:%M:%S")
        time_obj = time_obj-timedelta(hours=2)
        start_time.append(time_obj.strftime("%Y-%m-%d %H:%M:%S"))
with open("segment_end_time.txt","r") as f:
    for line in f.readlines():
        line=str.strip(line)
        time_obj = datetime.strptime(line, "%Y-%m-%d %H:%M:%S")
        time_obj = time_obj-timedelta(hours=2)
        end_time.append(time_obj.strftime("%Y-%m-%d %H:%M:%S"))
df = pd.read_csv("workload.csv")
print(df["timestamp"].dtype)
np_timestamp = df["timestamp"].to_numpy()
start_indexes = np.array([np.where(np_timestamp == x)[0][0] for x in start_time])
end_indexes = np.array([np.where(np_timestamp == x)[0][0] for x in end_time])
fig, ax = plt.subplots()
df.plot(x="timestamp", y="num_request", ax=ax)
for idx in start_indexes:
    ax.axvline(x=idx, color='r', alpha=0.5)
for idx in end_indexes:
    ax.axvline(x=idx, color='g', alpha=0.5)
plt.show()
