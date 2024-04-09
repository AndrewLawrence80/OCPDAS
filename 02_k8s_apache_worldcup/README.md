# 02 K8S Apache WorldCup

The directory `02_k8s_apache_worldcup` contains the code and results related to Apache performance benchmarking on K8S, serving as additional evidence for the approximate linear relation between workload and CPU utilization rate, consisting of the following files:

1. `benchmark.py`: Benchmark test program described in directory `01_k8s_apache_benchmark` with slightly differences, using real-world WorldCup 98 dataset.
2. `log`: Directory of benchmark test log organized in the following format:

   ```bash
   workload 3300, cpu utilization 5m
   62.37671971321106
   workload 3500, cpu utilization 21m
   62.29684543609619
   ```

   odd lines are cpu utilization logs, even lines are running time of the workload.

3. `plot_log.py`: Draw figures to show the relation between workload and CPU utilization rate
4. `img`: Directory containing plotted log.

## How to Use

1. Run `benchmark.py` and redirect the `stdout` to log file following:

    ```bash
        python benchmark.py > log/1998-06-10.log
    ```

    > **Note:** You can run multiple deployments to simultanously conduct the simualtions by modifying the apache.yaml and apply it to Minikube
2. Run `plot_log.py`.
