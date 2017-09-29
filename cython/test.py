import numpy as np
import multiply

a = np.arange(12, dtype=np.float64).reshape((3,4))
print(a)
multiply.multiply(a, 3)
print(a)
