from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Set
import math


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
    
    def __neg__(self) -> "Value":
        return self * -1

    def __sub__(self, other: "Value") -> "Value":
        return self + (-other)

    def __rsub__(self, other: "Value") -> "Value":
        return other + (-self)
    
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
    
    def __pow__(self, other: float) -> "Value":
        assert isinstance(other, (int, float))
        out = Value(self.data ** other)

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        out._prev = {self}
        out._op = f"**{other}"
        return out
    
    def tanh(self) -> "Value":
        t = math.tanh(self.data)
        out = Value(t)

        def _backward():
            # d/dx tanh(x) = 1 - tanh(x)^2
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        out._prev = {self}
        out._op = "tanh"
        return out
    
    def backward(self) -> None:
        topo = []
        visited = set()

        def build(v: "Value"):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()