from dataclasses import dataclass
from typing import Any

"""Implements an efficient last-in first-out Abstract Data Type using a Python List"""


@dataclass
class Stack:
    capacity: int  # capacity of stack

    def __post_init__(self) -> None:
        self.items = [None] * self.capacity  # array for stack
        self.num_items = 0  # number of items in stack

    def is_empty(self) -> bool:
        """Returns true if the stack self is empty and false otherwise"""
        return self.num_items == 0

    def is_full(self) -> bool:
        """Returns true if the stack self is full and false otherwise"""
        return self.num_items == self.capacity

    def push(self, item: Any) -> None:
        """Pushes item on the top of the Stack"""
        if self.num_items == self.capacity:
            raise IndexError
        self.items[self.num_items] = item
        self.num_items += 1

    def pop(self) -> Any:
        """Removes item from the top of the stack and returns it
        If stack is empty, raises IndexError"""
        if self.num_items == 0:
            raise IndexError
        self.num_items -= 1
        return self.items[self.num_items]

    def peek(self) -> Any:
        """Returns item on the top of the stack but does not remove it"""
        if self.num_items == 0:
            raise IndexError
        return self.items[self.num_items - 1]

    def size(self) -> int:
        """Returns the number of items in the stack. Must be O(1)"""
        return self.num_items

