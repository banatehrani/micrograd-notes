from __future__ import annotations

from graphviz import Digraph
from micrograd_notes.engine import Value


def trace(root: Value):
    nodes, edges = set(), set()

    def build(v: Value):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root: Value) -> Digraph:
    dot = Digraph(format="png", graph_attr={"rankdir": "LR"})

    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        label = f"{{ data {n.data:.4f} | grad {n.grad:.4f} }}"
        dot.node(name=uid, label=label, shape="record")

        if n._op:
            op_id = uid + n._op
            dot.node(name=op_id, label=n._op)
            dot.edge(op_id, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


if __name__ == "__main__":
    a = Value(2.0)
    b = Value(3.0)
    c = a * b + a
    c.backward()

    dot = draw_dot(c)
    dot.render("experiments/graph", cleanup=True)
    print("Wrote: experiments/graph.png")