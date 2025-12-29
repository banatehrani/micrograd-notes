from micrograd_notes.engine import Value

x = Value(3.0)
y = x ** 2
y.backward()

print("y:", y)          # expect data=9
print("x.grad:", x.grad) # expect 2*x = 6