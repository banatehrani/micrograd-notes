from micrograd_notes.engine import Value
import math

x = Value(2.0)
y = x.exp()
y.backward()

print("y.data:", y.data, " expected:", math.exp(2.0))
print("x.grad:", x.grad, " expected:", math.exp(2.0))