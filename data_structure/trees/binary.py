from typing import *


class BinaryTree:
    def __init__(self, value: int, max_len: int):
        self._content: List[int] = [value] + [0] * max_len

    def root_of(self):
        return 0
    def left_of(self, parent_idx: int) -> int:
        return parent_idx * 2 + 1

    def right_of(self, parent_idx: int) -> int:
        return parent_idx * 2 + 2

    def set_left(self, value: int, parent_idx: int):
        self._content[self.left_of(parent_idx)] = value

    def set_right(self, value: int, parent_idx: int):
        self._content[self.right_of(parent_idx)] = value

    def search_dfs_rec(self, val: int, current_idx: int, explored_idxs: Set[int]) -> int:
        print(f"exploring {current_idx}...")
        if current_idx in explored_idxs:
            return -1

        if self._content[current_idx] == val:
            return current_idx

        explored_idxs.add(current_idx)

        if not self.left_of(current_idx) > len(self._content):
            left_idx = self.search_dfs_rec(val, current_idx=self.left_of(current_idx), explored_idxs=explored_idxs)
            if left_idx >= 0:
                return left_idx

        if not self.right_of(current_idx) > len(self._content):
            right_idx = self.search_dfs_rec(val, current_idx=self.right_of(current_idx), explored_idxs=explored_idxs)
            if right_idx >= 0:
                return right_idx

        return -1

    def search_dfs(self, val: int) -> int:
        """
        search for value with dfs, return first list index
        :param val:
        :return:
        """
        return self.search_dfs_rec(val, self.root_of(), set())

    def print(self):
        print(self._content)


if __name__ == "__main__":
    tree: BinaryTree = BinaryTree(1, 10)

    tree.set_left(11, 0)
    tree.set_right(12, 0)

    tree.set_left(21, 1)
    tree.set_right(22, 1)

    tree.set_left(22, 2)
    tree.print()

    tree.search_dfs(1)
    print("----")
    print("got it on idx " + str(tree.search_dfs(11)))
    print("----")
    print("got it on idx " + str(tree.search_dfs(12)))

    print("----")
    print("got it on idx " + str(tree.search_dfs(21)))

    print("----")
    print("got it on idx " + str(tree.search_dfs(22)))