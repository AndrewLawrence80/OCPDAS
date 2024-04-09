import asyncio
import aiohttp
import time
from typing import Iterable
import pandas as pd
import subprocess
import json


class LoadGenerator:
    def __init__(self, url: str, concurrency: int, interval: float, timeout: float, workloads: Iterable[int], namespace: str, label: str) -> None:
        """
        Parameters
        ----------
        url: Target service url
        concurrency: How many requests are sent simultaneously
        interval: How long should a batch of workloads being processed,
            the per-second workload can be calculated using workloads[i]/interval
        timeout: A request is failed after timeout
        workloads: An array of request numbers, representing workload per interval,
            e.g., if workloads is set to [1000,2000,3000,4000,5000] and the interval is set to 60,
            the load generator will similuate workload from 1000/min to 5000/min, 
            and the simulation will take about 5min to finish.
        namespace: The target deployment namespace
        label: The target deployment label
        """
        self.url = url
        self.concurrency = concurrency
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
        t_sleep = self.interval/workload*self.concurrency
        success_count = 0
        failure_count = 0
        for _ in range(workload//self.concurrency+1):
            start_time = time.time()  # Record the start time of the batch
            async with aiohttp.ClientSession() as session:
                # Send multiple requests concurrently
                tasks = [self.send_request(session) for _ in range(self.concurrency)]
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
                print(f"Successful requests: {success_count}, Failed requests: {failure_count}")
        return failure_count

    def get_cpu_utilization(self):
        command = ["kubectl", "top", "pod", "--namespace", self.namespace, "-l", self.label, "--no-headers"]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout.split()[1]

    def run(self):
        r_cpu_dict = {}
        failure_dict = {}
        for workload in self.workloads:
            workload -= workload % self.concurrency
            failure_count = asyncio.run(self.gen_load(workload))
            r_cpu = self.get_cpu_utilization()
            print("workload %d, cpu utilization %s" % (workload, r_cpu))
            r_cpu_dict[workload] = r_cpu.rstrip("m")
            failure_dict[workload] = failure_count
        return r_cpu_dict, failure_dict


if __name__ == "__main__":
    # A benchmark test
    workload_candidates = range(5000, 40000, 2500)
    load_generator = LoadGenerator("http://192.168.49.2:30080", concurrency=100, interval=60, timeout=5, workloads=workload_candidates, namespace="ocpdas", label="app=apache")
    r_cpu_dict, failure_dict = load_generator.run()
    with open("r-cpu.json", "w") as f:
        json.dump(r_cpu_dict, f, indent=4)
    with open("failure.json", "w") as f:
        json.dump(failure_dict, f, indent=4)
