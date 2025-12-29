from micrograd_notes.engine import Value

a = Value(5.0)
b = Value(2.0)

c = a - b
c.backward()

print("c:", c)          # expect data=3
print("a.grad:", a.grad) # expect 1
print("b.grad:", b.grad) # expect -1