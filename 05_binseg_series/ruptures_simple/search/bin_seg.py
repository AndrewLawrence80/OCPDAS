from __future__ import annotations
from typing import List, Tuple
from ..cost.base_cost import BaseCost
from ..utils.pairwise import pairwise
import numpy as np
import warnings


class SegNode:
    def __init__(self, start: int, end: int, cost: int, left_child: SegNode = None, right_child: SegNode = None) -> None:
        """
        T[start,end) from time series T
        Params:
            start: start index of the segment
            end: end index of the segment
            cost: cost of the segment, denoting the homogeneity of the segment
            left_child: if the segment is constructed by merging two subsequence, then get the left child
            right_child: if the segment is constructed by merging two subsequence, then get the right child
        """
        self.start = start
        self.end = end
        self.cost = cost
        self.left_child = left_child
        self.right_child = right_child

    def get_gain(self):
        if self.left_child is None or self.right_child is None:
            raise ValueError("All child nodes cannot be None")
        return self.cost-(self.left_child.cost+self.right_child.cost)

    def __lt__(self, other: SegNode) -> bool:
        return self.get_gain() < other.get_gain()

    def __eq__(self, __value: SegNode) -> bool:
        return isinstance(__value, self.__class__) and self.start == __value.start and self.end == __value.end

    def __hash__(self):
        return hash((self.__class__, self.start, self.end))

    def __str__(self) -> str:
        return "[%d,%d)" % (self.start, self.end)


class SegNodeGainComparator:
    def __call__(self, node_a: SegNode, node_b: SegNode) -> bool:
        return node_a.get_gain() < node_b.get_gain()


class BinSeg:
    """Top-down segmentation."""

    def __init__(self, cost_model: BaseCost, init_seg_size: int = 1):
        """Initialize a BottomUp instance.

        Params:
            cost_model: segment model. Not used if `custom_cost` is not None.
            init_seg_size: initial segment length. Defaults to 1 sample.
        """
        self.cost_model = cost_model
        if init_seg_size != self.cost_model.get_min_size():
            warnings.warn("init_seg_size is not equals to the cose model required min_size, the search model will use the maximum from them")
        self.init_seg_size = max(init_seg_size, self.cost_model.get_min_size())
        self.n_samples = None
        self.signal = None

    def fit(self, signal: np.ndarray) -> BinSeg:
        """"
        Params:
            signal: signal to segment. Shape (n_samples, n_features) or (n_samples,).

        Returns:
            self
        """
        # update some params
        self.cost_model.fit(signal)
        if signal.ndim == 1:
            (n_samples,) = signal.shape
        else:
            n_samples, _ = signal.shape
        self.n_samples = n_samples
        return self

    def _dfs(self, start_idx: int, end_idx: int):
        # if start_idx>=end_idx:
        #     return None
        if end_idx-start_idx == 1:
            return SegNode(start_idx, end_idx, 0, None, None)
        current_cost = self.cost_model.error(start_idx, end_idx)
        current_parent = SegNode(start_idx, end_idx, current_cost, None, None)
        gain_list = [(bkp_idx, self.cost_model.error(start_idx, end_idx)-self.cost_model.error(start_idx, bkp_idx)-self.cost_model.error(bkp_idx, end_idx)) for bkp_idx in range(start_idx+1, end_idx)]
        bkp_idx, gain = max(gain_list, key=lambda x: x[1])
        current_parent.left_child = self._dfs(start_idx, bkp_idx)
        current_parent.right_child = self._dfs(bkp_idx, end_idx)
        return current_parent

    def bin_search(self):
        root_seg = self._dfs(0, self.n_samples)
        return root_seg
    
    def _dfs_early_stop(self,start_idx:int,end_idx:int):
        if end_idx-start_idx == 1:
            return SegNode(start_idx, end_idx, 0, None, None)
        current_cost = self.cost_model.error(start_idx, end_idx)
        current_parent = SegNode(start_idx, end_idx, current_cost, None, None)
        if current_cost < (end_idx-start_idx)*650000:
            return current_parent
        else:
            gain_list = [(bkp_idx, self.cost_model.error(start_idx, end_idx)-self.cost_model.error(start_idx, bkp_idx)-self.cost_model.error(bkp_idx, end_idx)) for bkp_idx in range(start_idx+1, end_idx)]
            bkp_idx, gain = max(gain_list, key=lambda x: x[1])
            current_parent.left_child = self._dfs(start_idx, bkp_idx)
            current_parent.right_child = self._dfs(bkp_idx, end_idx)
            return current_parent

    def bin_search_early_stop(self):
        root_seg = self._dfs_early_stop(0, self.n_samples)
        return root_seg