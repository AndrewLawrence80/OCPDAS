# 01 K8S Apache Benchmark

The directory `01_k8s_apache_benchmark` contains the code and results related with Apache performance benchmark on K8S, consisting of the following files:

1. `apache.yaml`: K8S Apache deployment for benchmark test in the background section of the paper.
2. `benchmark.py`: Benchmark test program, which tests the cpu utilization and failure rate of deployed apache server by sending increased requests to access the index page.
3. `r-cpu.json` and `failure.json`: Benchmark test results of the cpu utilzation and the number of timed-out reqeusts.
4. `plot_benchmark.py`: Draw the figure showing the approximate linear relation between workload and cpu utilization rate.
5. `benchmark.pdf`: The figure plotted by `plot_benchmark.py`.

> **Note:** The existing benchmark result is conducted using Minikube deployed on an Intel Core i7-9750H platform.

## How to Use

1. Make sure `Minikube` is correctly installed and running.
2. Make sure `kubectl` is correctly installed.
3. Apply the `apache.yaml` by the following command

   ``` bash
    kubectl apply -f apache.yaml
   ```

4. Run `benchmark.py` to get benchmark result.
5. Run `plot_benchmark.py` to get result figure.
