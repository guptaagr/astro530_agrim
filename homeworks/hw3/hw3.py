import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from scipy.special import expn

from astro530.integration import integrate_function
from astro530.planck import planck_wavenumber

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.formatter.useoffset": False,
})

# Problem 4: Emergent intensity integrator
# Linear source function
a0 = 1.0
a1 = 0.7

def S_linear(tau):
    return a0 + a1 * tau.value

def emergent_intensity_linear(mu, tau_max, dtau):
    def integrand(tau):
        return S_linear(tau) * np.exp(-tau.value / mu)
    return (1 / mu) * integrate_function(
        integrand,
        0.0 * u.dimensionless_unscaled,
        tau_max,
        dtau,
        method="trapz"
    )

mu_vals = np.linspace(0.05, 1.0, 200)
I_vals = np.array([
    emergent_intensity_linear(mu, 50.0 * u.dimensionless_unscaled, 1e-3 * u.dimensionless_unscaled)
    for mu in mu_vals
])

S_mu = a0 + a1 * mu_vals
frac_err = np.abs((I_vals - S_mu) / I_vals)

plt.figure(figsize=(7.5, 5.5))
plt.plot(mu_vals, I_vals, linewidth=2.5, label=r"$I_\nu^+(0,\mu)$ (numerical)")
plt.plot(mu_vals, S_mu, "--", linewidth=2.5, label=r"$S(\tau=\mu)$ (E--B)")
plt.xlabel(r"$\mu$")
plt.ylabel("Arbitrary units")
plt.legend()
plt.tight_layout()
plt.savefig("p4a_linear_I_vs_mu.svg")
plt.close()

plt.figure(figsize=(7.5, 5.5))
plt.plot(mu_vals, frac_err, linewidth=2.5)
plt.yscale("log")
plt.xlabel(r"$\mu$")
plt.ylabel(r"Fractional error $\left|\frac{I-S(\mu)}{I}\right|$")
plt.tight_layout()
plt.savefig("p4a_linear_frac_error.svg")
plt.close()

def max_error(tau_max, dtau):
    errs = []
    for mu in mu_vals:
        I = emergent_intensity_linear(mu, tau_max, dtau)
        errs.append(abs((I.value - (a0 + a1 * mu)) / I.value))
    return max(errs)

T0 = 5777 * u.K
nu_tilde = 2.0 / u.micron

def T_of_tau(tau):
    return T0 * (0.75 * tau.value + 0.5) ** 0.25

def S_realistic(tau):
    return planck_wavenumber(
        nu_tilde * np.ones_like(tau.value),
        T_of_tau(tau)
    )

def emergent_intensity_real(mu, tau_max, dtau):
    def integrand(tau):
        return S_realistic(tau) * np.exp(-tau.value / mu)
    return (1 / mu) * integrate_function(
        integrand,
        0.0 * u.dimensionless_unscaled,
        tau_max,
        dtau,
        method="trapz"
    )

I_real = u.Quantity([
    emergent_intensity_real(mu, 50.0 * u.dimensionless_unscaled, 1e-3 * u.dimensionless_unscaled)
    for mu in mu_vals
])

S_mu_real = planck_wavenumber(
    nu_tilde,
    T_of_tau(mu_vals * u.dimensionless_unscaled)
)

frac_err_real = np.abs((I_real - S_mu_real) / I_real).decompose().value

plt.figure(figsize=(7.5, 5.5))
plt.plot(mu_vals, frac_err_real, linewidth=2.5)
plt.yscale("log")
plt.xlabel(r"$\mu$")
plt.ylabel(r"Fractional error $\left|\frac{I-S(\mu)}{I}\right|$")
plt.tight_layout()
plt.savefig("p4c_realistic_frac_error.svg")
plt.close()

tau_plot = np.logspace(-3, 2, 500) * u.dimensionless_unscaled
scale = 1e-7
S_plot = (S_realistic(tau_plot).value / scale)
plt.figure(figsize=(7.5, 5.5))
plt.plot(tau_plot.value, S_plot, linewidth=2.5)
plt.xscale("log")
plt.xlabel(r"$\tau$")
plt.ylabel(r"$S_\nu(\tau)\ \times 10^{-7}\ \mathrm{(arbitrary\ units)}$")
plt.tight_layout()
plt.savefig("p4c_S_vs_tau.svg")
plt.close()

#Problem 5: Exponential integrals
def En_of_x(x, n):
    x = np.asarray(x)
    return expn(n, x)

def integrate_En(n, xmin, xmax, N, tail_correct=True):
    ln_x = np.linspace(np.log(xmin), np.log(xmax), N)
    x = np.exp(ln_x)
    y = expn(n, x)
    integral = np.trapz(y * x, ln_x)
    if tail_correct:
        integral += expn(n + 1, xmax)
    return integral

def frac_error(calc, truth):
    return abs(1.0 - calc / truth)

x = np.logspace(-6, 2, 500)

plt.figure()
plt.loglog(x, expn(1, x), label=r"$E_1(x)$")
plt.loglog(x, expn(2, x), label=r"$E_2(x)$")
plt.loglog(x, expn(3, x), label=r"$E_3(x)$")
plt.xlabel(r"$x$")
plt.ylabel(r"$E_n(x)$")
plt.legend()
plt.tight_layout()
plt.savefig("expn_functions.svg")
plt.close()

for n in [1, 2, 3]:
    truth = 1.0 / n
    val = integrate_En(n, xmin=1e-12, xmax=200.0, N=300)
    print(n, val, frac_error(val, truth))

n = 1
truth = 1.0

N_list = np.array([50, 100, 200, 300, 400, 600, 800])
errs = []

for N in N_list:
    val = integrate_En(n, xmin=1e-12, xmax=200.0, N=N)
    errs.append(frac_error(val, truth))

plt.figure()
plt.loglog(N_list, errs)
plt.xlabel(r"$N$")
plt.ylabel("Fractional error")
plt.tight_layout()
plt.savefig("precision_vs_N.svg")
plt.close()

xmax_list = 10.0**np.linspace(0, 3, 10)
errs = []

for xmax in xmax_list:
    val = integrate_En(2, xmin=1e-12, xmax=xmax, N=300)
    errs.append(frac_error(val, 0.5))

plt.figure()
plt.loglog(xmax_list, errs)
plt.xlabel(r"$x_{\max}$")
plt.ylabel("Fractional error")
plt.tight_layout()
plt.savefig("precision_vs_xmax.svg")
plt.close()