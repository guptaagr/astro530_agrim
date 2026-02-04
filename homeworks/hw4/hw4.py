import numpy as np
from astropy import units as u
from scipy.special import expn

from astro530.integration import trapz

def H_nu_surface(tau, S_func):
    tau = u.Quantity(tau).to(u.dimensionless_unscaled)
    S = S_func(tau)
    E2 = expn(2, tau.value)
    integrand = S * E2
    return 0.5 * trapz(tau, integrand)

a0 = 1.2 * u.dimensionless_unscaled
a1 = 0.8 * u.dimensionless_unscaled
a2 = 0.3 * u.dimensionless_unscaled

def S_quadratic(tau):
    return a0 + a1 * tau + a2 * tau**2

tau_min = 1e-6 * u.dimensionless_unscaled
tau_max = 100.0 * u.dimensionless_unscaled
N = 2000
tau_grid = np.logspace(np.log10(tau_min.value), np.log10(tau_max.value), N) * u.dimensionless_unscaled

H0_quad = H_nu_surface(tau_grid, S_quadratic)
print("H_nu(0) (quadratic) =", H0_quad)

a0L = 2.0 * u.dimensionless_unscaled
a1L = 3.0 * u.dimensionless_unscaled

def S_linear(tau):
    return a0L + a1L * tau

H0_lin_num = H_nu_surface(tau_grid, S_linear)
H0_lin_true = a0L/4 + a1L/6

print("H_nu(0) numeric =", H0_lin_num)
print("H_nu(0) analytic =", H0_lin_true)
print("fractional error =", abs((H0_lin_num - H0_lin_true) / H0_lin_true).decompose().value)