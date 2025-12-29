from micrograd_notes.engine import Value

x = Value(2.0)
y = x.tanh()
y.backward()

print("y:", y)          # ~0.9640
print("x.grad:", x.grad) # ~0.0707