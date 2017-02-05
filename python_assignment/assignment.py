import numpy as np

#a = np.arange(3,5)
a = [3, 4]
b = a
c = a[:]
d = a.copy()

print(a is b) # True
print(a is c) # False
print(a is d) # False

print(a, b, c, d) #[3 4] [3 4] [3 4] [3 4]

a[0] = -11

print(a, b, c, d) #[-11   4] [-11   4] [-11   4] [3 4]
