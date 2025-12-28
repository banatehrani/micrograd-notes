from micrograd_notes.engine import Value

a = Value(2.0)
b = Value(3.0)
c = a * b + a  # c = a*b + a

c.backward()

print("c:", c)        # data should be 8.0
print("a.grad:", a.grad)  # should be b + 1 = 4.0
print("b.grad:", b.grad)  # should be a = 2.0