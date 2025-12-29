from micrograd_notes.engine import Value

a = Value(3.0)

x = 2 + a
y = 2 * a
z = (2 * a + 1).tanh()
z.backward()

print("x:", x)  # data=5
print("y:", y)  # data=6
print("a.grad:", a.grad)  # should be non-zero