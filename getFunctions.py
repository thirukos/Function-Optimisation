import numpy as np
import sympy

# function1: 8*(x-9)^4+6*(y-1)^2
# function2: Max(x-9,0)+6*|y-1|

def getFunction (func_num):
    x,y = sympy.symbols('x,y', real=True)
    if func_num == 1:
        func = (8*((x-9)**4))+(6*((y-1)**2))
        dfdx = sympy.diff(func,x)
        dfdy = sympy.diff(func,y)
        print(f'function: {func}')
        # func = sympy.lambdify((x, y), func)
        # dfdx = sympy.lambdify((x), dfdx)
        # dfdy = sympy.lambdify((y), dfdy)
        func = lambda x, y: (8*((x-9)**4))+(6*((y-1)**2))
        dfdx = lambda x: 32*(x-9)**3
        dfdy = lambda y: 12*y - 12
        return func, (dfdx, dfdy)
    elif (func_num == 2):
        f = sympy.Max(x-9, 0) + 6*sympy.Abs(y-1)
        func = lambda x, y: np.maximum(x-9, 0) + 6*np.abs(y-1)
        dfdx = lambda x: np.heaviside(x-9, 0)
        dfdy = lambda y: 6 * np.sign(y-1)
        print(f'function: {f}')
        return func, (dfdx, dfdy)
    else: 
        print('Only 2 functions available, select between 1 and 2')


