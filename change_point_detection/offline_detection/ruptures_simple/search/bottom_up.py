from __future__ import annotations
from typing import List, Tuple
from ..cost.base_cost import BaseCost
from ..utils.heap import MinHeap
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


class BottomUp:
    """Bottom-up segmentation."""

    def __init__(self, cost_model: BaseCost, init_seg_size: int = 1):
        """Initialize a BottomUp instance.

        Params:
            cost_model: segment model. Not used if ``'custom_cost'`` is not None.
            init_seg_size: initial segment length. Defaults to 1 sample.
        """
        self.cost_model = cost_model
        if init_seg_size != self.cost_model.get_min_size():
            warnings.warn("init_seg_size is not equals to the cose model required min_size, the search model will use the maximum from them")
        self.init_seg_size = max(init_seg_size, self.cost_model.get_min_size())
        self.n_samples = None
        self.signal = None

    def fit(self, signal: np.ndarray) -> BottomUp:
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

    def _init_leaves(self) -> List[Tuple[int, int]]:
        partition = []
        for idx in range(0, self.n_samples-self.init_seg_size+1, self.init_seg_size):
            partition.append((idx, idx+self.init_seg_size))

        partition = sorted(partition, key=lambda x: x[0])
        # compute segment costs
        leaves = []
        for start, end in partition:
            cost = self.cost_model.error(start, end)
            leaf = SegNode(start, end, cost)
            leaves.append(leaf)
        return leaves

    def _merge_seg(self, left_child: SegNode, right_child: SegNode) -> SegNode:
        start, end = left_child.start, right_child.end
        cost = self.cost_model.error(start, end)
        parent = SegNode(start, end, cost, left_child, right_child)
        return parent

    def merge_search(self):
        leaves = self._init_leaves()
        heap = MinHeap(SegNodeGainComparator())
        for left_child, right_child in pairwise(leaves):
            heap.heap_push(self._merge_seg(left_child, right_child))

        root_seg = None
        while heap.get_size() > 0:
            merged_seg: SegNode = heap.heap_pop()
            if heap.get_size() == 0:
                root_seg = merged_seg
                break

            new_candidates = []

            idx = -1
            while idx < heap.get_size()-1:
                idx += 1
                if merged_seg.left_child == heap[idx].right_child:
                    candidate = self._merge_seg(heap[idx].left_child, merged_seg)
                    new_candidates.append(candidate)
                    heap.inner_arr_pop(idx)
                    idx -= 1
                    continue
                if merged_seg.right_child == heap[idx].left_child:
                    candidate = self._merge_seg(merged_seg, heap[idx].right_child)
                    new_candidates.append(candidate)
                    heap.inner_arr_pop(idx)
                    idx -= 1
                    continue

            for candidate in new_candidates:
                heap.inner_arr_append(candidate)

            heap.heapify()

        return root_seg
