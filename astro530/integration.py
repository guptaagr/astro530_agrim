import numpy as np
from astropy import units as u

def trapz(x, y):
    x = u.Quantity(x)
    y = u.Quantity(y)
    dx = np.diff(x)
    ymid = 0.5 * (y[:-1] + y[1:])
    return np.sum(ymid * dx)

def box(x, y):
    x = u.Quantity(x)
    y = u.Quantity(y)
    dx = np.diff(x)
    return np.sum(y[:-1] * dx)

def integrate_function(func, xmin, xmax, dx, method="trapz"):
    xmin = u.Quantity(xmin)
    xmax = u.Quantity(xmax)
    dx = u.Quantity(dx)

    n = int(np.ceil(((xmax - xmin) / dx).decompose().value)) + 1
    x = xmin + np.arange(n) * dx
    x = x[x <= xmax]
    y = func(x)

    if method == "trapz":
        return trapz(x, y)
    elif method == "box":
        return box(x, y)
    else:
        raise ValueError("method must be 'trapz' or 'box'")