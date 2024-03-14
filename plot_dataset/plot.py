import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os

font = {'size': 16}

matplotlib.rc('font', **font)

data_path = "dataset"
color = "#3f51b5" # material indigo
for csv in os.listdir(data_path):
    date_str = csv.split(".")[0]
    df = pd.read_csv(os.path.join(data_path, csv))
    df["timestamp"] = df["timestamp"].apply(lambda s: s.split(" ")[1])
    df["request number scaled by 10x"]=df["num_request"]/10
    ax = df.plot(x="timestamp", y="request number scaled by 10x", xlabel="time (min)", ylabel="request number (x10 requests)", color=color,figsize=(16,9))
    ax.grid(True,linestyle="--")
    ax.set_title(date_str.replace("_"," "))
    ax = plt.gca()
    # Remove the upper and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig("plot_dataset/img/"+date_str+".pdf")
    plt.close()
