from micrograd_notes.engine import Value
from micrograd_notes.nn import MLP

import random
random.seed(42)

# Tiny dataset (Karpathy style)
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

mlp = MLP(3, [4, 4, 1])
lr = 0.05

for k in range(50):
    # forward
    ypred = [mlp([Value(xi) for xi in x]) for x in xs]

    # loss: sum of squared error
    loss = sum((yout - ygt) ** 2 for yout, ygt in zip(ypred, ys))

    # backward
    mlp.zero_grad()
    loss.backward()

    # update
    for p in mlp.parameters():
        p.data += -lr * p.grad

    if k % 5 == 0 or k == 49:
        print(f"iter={k:02d} loss={loss.data:.6f}")


print("\nfinal predictions:")
for x, ygt in zip(xs, ys):
    y = mlp([Value(xi) for xi in x])
    print(f"x={x} pred={y.data:.4f} target={ygt:+.1f}")