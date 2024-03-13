from typing import Callable, Any, List


class MinHeap:
    def __init__(self, comparator: Callable[[Any, Any], bool]) -> None:
        """
        Heap implementation referring to https://algs4.cs.princeton.edu/24pq/
        Parameters
        ----------
        comparator: Callable
            Defines how to compare elements in the MinHeap,
            which should return true when call ```comparator(elemant_a, element_b)```
            if element_a < element_b, false on the contrary
        """
        self.arr = []
        self.size = 0
        self.comparator = comparator

    def _swap(self, index_a: int, index_b: int) -> None:
        temp = self.arr[index_a]
        self.arr[index_a] = self.arr[index_b]
        self.arr[index_b] = temp

    def _get_left_index(self, index: int) -> int:
        return 2*index+1

    def _get_right_index(self, index) -> int:
        return 2*index+2

    def _get_parent_index(self, index) -> int:
        return (index-1)//2

    def _sink(self, index) -> None:
        while index*2+1 < self.size:
            min_index = index
            left_index = self._get_left_index(index)
            right_index = self._get_right_index(index)
            if left_index < self.size and self.comparator(self.arr[left_index], self.arr[min_index]):
                min_index = left_index
            if right_index < self.size and self.comparator(self.arr[right_index], self.arr[min_index]):
                min_index = right_index
            if min_index != index:
                self._swap(min_index, index)
                index = min_index
            else:
                break

    def _swim(self, index) -> None:
        while index > 0:
            parent_index = self._get_parent_index(index)
            if self.comparator(self.arr[index], self.arr[parent_index]):
                self._swap(index, parent_index)
                index = parent_index
            else:
                break

    def heapify(self) -> None:
        for index in reversed(range(self.size//2)):
            self._sink(index)

    def heap_peek(self) -> Any:
        if self.size <= 0:
            raise IndexError("Heap is already empty, current size %d" % self.size)
        return self.arr[0]

    def heap_pop(self) -> Any:
        if self.size <= 0:
            raise IndexError("Heap is already empty, current size %d" % self.size)
        top = self.arr[0]
        self._swap(0, self.size-1)
        self.size -= 1
        self._sink(0)
        return top

    def heap_pop_by_index(self, index: int) -> None:
        if self.size <= 0:
            raise IndexError("Heap is already empty, current size %d" % self.size)
        if index != self.size-1:
            self._swap(index, self.size-1)
            self.size -= 1
            self._swim(index)
            self._sink(index)
        else:
            self.size -= 1

    def heap_push(self, item: Any) -> None:
        if self.size == len(self.arr):
            self.arr.append(item)  # Ensure size increase by 1
        else:
            self.arr[self.size] = item
        self.size += 1
        index = self.size-1
        self._swim(index)

    def is_heap_empty(self) -> bool:
        return self.size == 0

    def __getitem__(self, index) -> Any:
        if index >= self.size:
            raise IndexError("Index out of range")
        return self.arr[index]

    def get_size(self) -> int:
        return self.size

    def clear(self) -> None:
        self.arr.clear()
        self.size = 0

    def inner_arr_append(self, item: Any) -> None:
        """
        append an item to the last of `self.arr`,
        `heapify` must be called later to rebuild heap
        """
        if self.size == len(self.arr):
            self.arr.append(item)
        else:
            self.arr[self.size] = item
        self.size += 1

    def inner_arr_pop(self, index: int) -> None:
        """
        pop an item at specified index from arr
        `heapify` must be called later to rebuild heap
        """
        if index >= self.size:
            raise IndexError("index out of range")
        self.arr.pop(index)
        self.size -= 1
