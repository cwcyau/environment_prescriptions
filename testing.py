import pytensor
pytensor.config.mode = "NUMBA"
from pytensor import tensor as pt
from pytensor.compile.function import function  # THIS imports the actual callable

print("PyTensor mode:", pytensor.config.mode)  # should be 'NUMBA'

# define simple function
x = pt.dscalar("x")
y = x ** 2
f = function([x], y)
print("f(5) =", f(5))