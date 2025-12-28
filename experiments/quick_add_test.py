from micrograd_notes.engine import Value

a = Value(2.0)
b = Value(3.0)
c = a + b

print(c)  # expect Value(data=5.0, grad=0.0)

d = a * b

print(d)  # expect Value(data=6.0, grad=0.0)