# 08 Ablation Experiments

The directory `08_ablation_experiments` contains code and results related to running ablation experiments to compare change point detection performance among t-tests with fixed warmup windows, bare reinforcement learning, and t-tests with warmup windows guided by reinforcement learning. It proves the effectiveness of t-tests with warmup windows guided by reinforcement learning, which achieve the highest F1 score and slightly lower missing detection rate. The directory consists of the following files:

1. `generate_synthetic_example.ipynb`: Code to generate synthetic data examples. The results will be stored in `signal_example` and `signal_img_example`.
2. `generate_synthetic_data.ipynb`: Code to generate synthetic data. The results will be stored in `signal` and `signal_img`.
3. `t_test_cpd.py`: Core code of t-test change point detection.
4. `model.py`, `agent.py`, `replay_buffer.py`, `train_utils.py`, `world_cup_env.py`: Supporting code to run reinforcement learning.
5. `run_rl.py`: Code to run reinforcement learning on synthetic datasets to learn prior knowledge when changes are about to occur.
6. `cpd_evaluation.py`: Change point detection (CPD) evaluation utils, criteria include precision, recall, F1 score, average detection delay, and [NAB score](http://arxiv.org/abs/1510.03336).
7. `eval_t_test_fixed_warmup.ipynb`: Code to evaluate the CPD performance of t-tests with fixed warmup windows. The output will be stored in `eval_fixed`.
8. `eval_rl_cpd.ipynb`: Code to evaluate the CPD performance of bare reinforcement learning. The output will be stored in `eval_rlcpd`.
9. `eval_t_test_dynamic_warmup.ipynb`: Code to evaluate the CPD performance of t-tests with dynamic warmup windows guided by RL agents. The output will be stored in `eval_dynamic`.

## How to use

1. If you want to see the synthetic example, run `generate_synthetic_example.ipynb`.
2. If you want to regenerate synthetic data, run `generate_synthetic_data.ipynb`. Then proceed to step 3.
3. If you want to train an RL agent on the synthetic data, run `run_rl.ipynb`.
4. Run `eval_t_test_fixed_warmup.ipynb`, `eval_rl_cpd.ipynb`, `eval_t_test_dynamic_warmup.ipynb` to see the evaluation results. The results per signal may be different from those displayed in the original paper after retraining the RL agent, but the overall performance will not suffer high deviations.
