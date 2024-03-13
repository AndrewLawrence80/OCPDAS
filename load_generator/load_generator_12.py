import asyncio
import aiohttp
import time
from typing import Iterable
import pandas as pd
import subprocess
from kubernetes import client, config


class LoadGenerator:
    def __init__(self, url: str, batch_size: int, interval: float, timeout: float, workloads: Iterable[int], namespace: str, label: str) -> None:
        self.url = url
        self.batch_size = batch_size
        self.interval = interval
        self.timeout = timeout
        self.workloads = workloads
        self.namespace = namespace
        self.label = label

    async def send_request(self, session: aiohttp.ClientSession):
        try:
            async with session.get(self.url, timeout=self.timeout) as response:
                # Print the status code of the response
                status_code = response.status
                return status_code
        except asyncio.TimeoutError:
            return "timeout"
        except Exception as e:
            return f"error: {e}"

    async def gen_load(self, workload: int):
        t_sleep = self.interval/workload*self.batch_size
        success_count = 0
        failure_count = 0
        for _ in range(workload//self.batch_size+1):
            start_time = time.time()  # Record the start time of the batch
            async with aiohttp.ClientSession() as session:
                # Send multiple requests concurrently
                tasks = [self.send_request(session) for _ in range(self.batch_size)]
                done, _ = await asyncio.wait(tasks)

                # Count the number of successful and failed requests
                for task in done:
                    result = await task
                    if isinstance(result, int) and result == 200:
                        success_count += 1
                    else:
                        failure_count += 1

                # Calculate the time elapsed since the start of the batch
                elapsed_time = time.time() - start_time

                # Adjust the sleep interval to maintain the desired request rate
                if elapsed_time < t_sleep:
                    await asyncio.sleep(t_sleep - elapsed_time)

                # Print the counts
                # print(f"Successful requests: {success_count}, Failed requests: {failure_count}")

    def get_cpu_utilization(self):
        command = ["kubectl", "top", "pod", "--namespace", self.namespace, "-l", self.label, "--no-headers"]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout.split()[1]

    def run(self):
        start_time = time.time()
        for workload in self.workloads:
            workload -= workload % self.batch_size
            asyncio.run(self.gen_load(workload))
            r_cpu = self.get_cpu_utilization()
            print("workload %d, cpu utilization %s" % (workload, r_cpu))
        print(str(time.time()-start_time))


if __name__ == "__main__":
    df = pd.read_csv("dataset/workload_1998-06-22.csv")
    df["num_request"] = df["num_request"].apply(lambda x: x//10)
    workloads = df["num_request"].to_numpy()
    for workload in workloads:
        load_generator = LoadGenerator("http://192.168.49.2:30080", batch_size=100, interval=60, timeout=5, workloads=[workload], namespace="autoscaling", label="app=apache-0")
        load_generator.run()
