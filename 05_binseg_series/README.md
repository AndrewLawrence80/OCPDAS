# 05 Binseg Series

The directory `05_binseg_series` contains code and results related to running time series segmentation using the Binseg algorithm on simulated random data and the WorldCup dataset. It consists of the following files:

1. `ruptures_simple`: Source code of the Binseg algorithm derived from the Python package `ruptures`. The algorithm will generate a complete binary segment tree until the leaf segment length is 1. This is because the number of changes, $k$, is unknown.
2. `tree_utils.py`: Some basic algorithms on trees.
3. `binseg_example.ipynb`: Code to run Binseg on randomly generated data.
4. `binseg_tree_example`: Results of the segment tree of `binseg_example.ipynb`.
5. `binseg_cpd_example`: Results of change point detection of `binseg_example.ipynb`.
6. `binseg_worldcup.ipynb`: Code to run Binseg on the WorldCup dataset.
7. `binseg_tree_worldcup`: Results of the segment tree of `binseg_worldcup.ipynb`.
8. `binseg_cpd_worldcup`: Results of change point detection of `binseg_worldcup.ipynb`.

## How to use

1. Run `binseg_example.ipynb`.
2. Run `binseg_worldcup.ipynb`.