import pandas as pd
df = pd.read_csv("workload.csv")
print(df.head(10))
df = df[["timestamp", "num_request", "size"]]
df.to_csv("tmp.csv", index=False)
