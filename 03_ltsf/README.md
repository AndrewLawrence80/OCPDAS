# 03 LTSF

The directory `03_ltsf` contains code and results related to a comparison between conventional offline-online training and sliding-window-based online training approaches in time series forecasting. The directory consists of the following files:

1. `model`: Directory with vanilla LSTM and some SOTA time series forecasting models, including LSTM, [NLinear](https://github.com/cure-lab/LTSF-Linear), [SegRNN](https://github.com/lss-1138/SegRNN), and [PatchTST](https://github.com/yuqinie98/PatchTST/).
2. `offline_training`: Directory with conventional offline-online training pipeline and training results.
3. `online_training`: Directory with sliding-window-based online training pipeline and training results.
4. `comparison.ipynb`: Notebook to compare the performance between step 2 and step 3.
5. `img_comparison`: Figures plotted by `comparison.ipynb`, showing the comparison results clearly.

## How to use

1. Go to the `offline_training` directory, run `run_exp.ipynb`, then run `run_evaluation.ipynb`. The file `run_exp.ipynb` will execute the training process on the everyday data. The file `run_evaluation.ipynb` will output the evaluation results using MAE as criteria.
2. Go to the `online_training` directory and do the same thing as above.
3. Run `comparison.ipynb`. The results will be outputted to `img_comparison`.