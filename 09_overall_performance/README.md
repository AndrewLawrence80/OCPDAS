# 09 Overall Performance

The directory `09_overall_performance` contains code and results related to a comparison of the overall performance among different autoscaling strategies, including reactive, proactive, and ocpdas. The performance is evaluated using criteria including constraints violation rate, scaling action numbers, and pod flapping counts. The directory consists of the following files:

1. `flapping_example.ipynb`: Code to demonstrate the pod flapping problem in K8S clearly. The output will be stored in `flapping_example`.
2. `reactive.ipynb`, `proactive.ipynb`, `ocpdas.ipynb`: Code to run simulation experiments of autoscaling using strategies of reactive, proactive, and ocpdas. The results will be stored in `reactive_result`, `proactive_result`, and `ocpdas_result`.
3. `flapping_analysis.ipynb`: Code to summarize the overall flapping count during autoscaling using the above 3 different strategies.
4. `result_to_csv.ipynb`: Convert JSON-formatted evaluation results into CSV format.

## How to use

1. If you want to see the flapping example, run `flapping_example.ipynb`.
2. If you want to reproduce the simulation experiment and check the evaluation results, run `reactive.ipynb`, `proactive.ipynb`, `ocpdas.ipynb`, `flapping_analysis.ipynb`, `result_to_csv.ipynb` sequentially. The results per workload may be different from those displayed in the original paper for retraining in `07_t_test_dynamic_warmup`, but the overall performance will not suffer high deviations.
