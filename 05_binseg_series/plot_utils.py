from ruptures_simple.search import SegNode
import graphviz
from collections import deque


def plot_seg_tree(root: SegNode) -> graphviz.Digraph:
    graph = graphviz.Digraph()
    graph.graph_attr['size'] = '9,3'
    seg_deque = deque([root])
    while len(seg_deque) > 0:
        parent_seg = seg_deque.popleft()
        if parent_seg.left_child is None and parent_seg.right_child is None:
            continue
        graph.node(str(parent_seg))
        left_child = parent_seg.left_child
        right_child = parent_seg.right_child
        graph.node(str(left_child))
        graph.node(str(right_child))
        graph.edge(str(parent_seg), str(left_child), label="%.2f" % parent_seg.get_gain())
        graph.edge(str(parent_seg), str(right_child), label="%.2f" % parent_seg.get_gain())
        seg_deque.append(left_child)
        seg_deque.append(right_child)
    graph.graph_attr['size'] = '9,3'
    return graph
