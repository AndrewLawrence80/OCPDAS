from ruptures_simple.search import SegNode
from typing import List, Dict
from collections import deque


def child_sort(root: SegNode):
    seg_deque = deque([])
    seg_deque.append(root)
    while len(seg_deque) > 0:
        current = seg_deque.popleft()
        left_child = current.left_child
        right_child = current.right_child
        if left_child is None and right_child is None:
            continue
        else:
            if left_child.start > right_child.start:
                current.right_child = left_child
                current.left_child = right_child


def preorder_dfs(current: SegNode, history: List[SegNode], path: List[SegNode], path_dict: Dict[SegNode, List[SegNode]]):
    if current.left_child is None and current.right_child is None:
        path.append(current)
        path_dict[current] = path.copy()
        path.pop()
        return
    history.append(current)
    path.append(current)
    preorder_dfs(current.left_child, history, path, path_dict)
    preorder_dfs(current.right_child, history, path, path_dict)
    path.pop()


def preorder_traverse(root: SegNode):
    history: List[SegNode] = []
    path: List[SegNode] = []
    path_dict: Dict[SegNode, List[SegNode]] = {}
    preorder_dfs(root, history, path, path_dict)
    return history, path_dict


def inorder_dfs(current: SegNode, history: List[SegNode], path: List[SegNode], path_dict: Dict[SegNode, List[SegNode]]):
    if current.left_child is None and current.right_child is None:
        path.append(current)
        path_dict[current] = path.copy()
        path.pop()
        return
    path.append(current)
    inorder_dfs(current.left_child, history, path, path_dict)
    history.append(current)
    inorder_dfs(current.right_child, history, path, path_dict)
    path.pop()


def inorder_traverse(root: SegNode, path: List, path_dict: Dict):
    history: List[SegNode] = []
    path: List[SegNode] = []
    path_dict: Dict[SegNode, List[SegNode]] = {}
    inorder_dfs(root, history, path, path_dict)
    return history, path_dict


def postorder_dfs(current: SegNode, history: List[SegNode], path: List[SegNode], path_dict: Dict[SegNode, List[SegNode]]):
    if current.left_child is None and current.right_child is None:
        path.append(current)
        path_dict[current] = path.copy()
        path.pop()
        return
    path.append(current)
    postorder_dfs(current.left_child, history, path, path_dict)
    postorder_dfs(current.right_child, history, path, path_dict)
    history.append(current)
    path.pop()


def postorder_traverse(root: SegNode):
    history: List[SegNode] = []
    path: List[SegNode] = []
    path_dict: Dict[SegNode, List[SegNode]] = {}
    postorder_dfs(root, history, path, path_dict)
    return history, path_dict
