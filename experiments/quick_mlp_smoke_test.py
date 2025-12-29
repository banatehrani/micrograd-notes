from micrograd_notes.engine import Value
from micrograd_notes.nn import MLP

mlp = MLP(3, [4, 4, 1])

x = [Value(2.0), Value(-1.0), Value(0.5)]
y = mlp(x)

print("y:", y)
print("num params:", len(mlp.parameters()))

y.backward()
print("first param grad:", mlp.parameters()[0].grad)