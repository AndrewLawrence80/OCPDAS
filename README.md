# OCPDAS Support Code

## What I have done

1. Shuffle WorldCup 98 dataset. From 1998-06-10 to 1998-07-08, about 2 matches were held every day from 15:30 to about 22:00 (localtime, maybe Paris time? I don't remeber clearly, but it doesn't really matter). The workload is the number of every-miniute requests parsing from nginx log. I removed some anomaly data points (At 22:00 every day the request number suddenly decreases, which may be caused by the restarting of a part of servers in cluster. And there are some rare burstive datapoints which are caused by unkown reasons. The whole preprocess code will be publish in another repo.)
2. Bench mark test on apache-httpd in k8s, which can be found in `apache-httpd` and `load_generator`.
3. Time series forecasting code in `ltsf`.
4. Online CPD using probability-based CUSUM algorithm, which can be found in `cpd_trash`. The folder is trashed.
5. Offline time series segmentation using bottom-up search in rupture, which can be found in `change_point_detection/offline_detection`.
6. Online dynamic warm-up windows selection learning method in DL and RL in `warm_up_selection`. The overall training process is still suffer from class imbalance. Negative sampling will come later.
7. Online CPD consists of 4 and the result of 6 which can be found in `ocpd`.
8. Simulation of reactive, proactive and OCPDAS and relative results, which can be found in `ocpdas_simulation`.

## Todos

1. Code refactoring, source code in the repo is currently a pile of grabage.
2. Code comments, which will come with refactoring.
3. A deployable framework, which may take about 3 to 6 month to program if I continue to work on the paper.

## Reproduction Environment

The reproduction environment requests are exported from my laptop using `conda`, feel free to open issue or email me if you run into trouble in environment problem. A clean requirements file will be provided later after I make these mess tidy. I'm really busy to catch the ddl currently QAQ.
