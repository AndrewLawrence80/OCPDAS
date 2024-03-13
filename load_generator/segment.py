import pandas as pd
from datetime import datetime, timedelta
import numpy as np
workload_df = pd.read_csv("workload.csv")
timestamp_np = workload_df["timestamp"].to_numpy()

start_time = []
with open("segment_start_time.txt", "r") as f:
    for line in f.readlines():
        line = str.strip(line)
        time_obj = datetime.strptime(line, "%Y-%m-%d %H:%M:%S")
        time_obj = time_obj-timedelta(hours=2)
        start_time.append(time_obj.strftime("%Y-%m-%d %H:%M:%S"))

end_time = []
with open("segment_end_time.txt", "r") as f:
    for line in f.readlines():
        line = str.strip(line)
        time_obj = datetime.strptime(line, "%Y-%m-%d %H:%M:%S")
        time_obj = time_obj-timedelta(hours=2)
        end_time.append(time_obj.strftime("%Y-%m-%d %H:%M:%S"))

print(len(start_time))
print(len(end_time))

start_time_indexes = np.array([np.where(timestamp_np == x)[0] for x in start_time]).squeeze()
end_time_indexes = np.array([np.where(timestamp_np == x)[0] for x in end_time]).squeeze()

segment_pairs = [(start_time_indexes[i], end_time_indexes[i]) for i in range(len(start_time_indexes))]

for i in range(len(segment_pairs)):
    seg_pair = segment_pairs[i]
    workload_segment = workload_df[seg_pair[0]:seg_pair[1]]
    workload_segment.to_csv("workload_"+workload_segment["timestamp"].to_numpy()[0].split(" ")[0]+".csv", index=False)
