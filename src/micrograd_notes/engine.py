from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Set


@dataclass
class Value:
    data: float
    grad: float = 0.0

    _backward: Callable[[], None] = field(default=lambda: None, repr=False)
    _prev: Set["Value"] = field(default_factory=set, repr=False)
    _op: str = field(default="", repr=False)

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"
