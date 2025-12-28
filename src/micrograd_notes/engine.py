from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Set


@dataclass(eq=False)
class Value:
    data: float
    grad: float = 0.0

    _backward: Callable[[], None] = field(default=lambda: None, repr=False)
    _prev: Set["Value"] = field(default_factory=set, repr=False)
    _op: str = field(default="", repr=False)

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"
    
    def __add__(self, other: "Value") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data)

        def _backward():
            # d(out)/d(self) = 1, d(out)/d(other) = 1
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        out._prev = {self, other}
        out._op = "+"
        return out
    
    def __mul__(self, other: "Value") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data)

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        out._prev = {self, other}
        out._op = "*"
        return out