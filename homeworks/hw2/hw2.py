import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.constants import sigma_sb, c

from astro530.planck import planck_wavenumber
from astro530.integration import integrate_function

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.formatter.useoffset": False,
})

#(b) Precision of calculation
T = 7500 * u.K

def truth_integral_wavenumber(T):
    return (sigma_sb / np.pi / c) * T**4 / u.sr

def B_of_nutilde(nutilde):
    return planck_wavenumber(nutilde, T)

unit_int = u.W * u.s / (u.m**3 * u.sr)

def precision(calc, truth):
    calc = calc.to(unit_int)
    truth = truth.to(unit_int)
    return abs(1 - (calc / truth).decompose().value)

truth = truth_integral_wavenumber(T)

nu_min_fixed = 1e-6 / u.micron
nu_max_fixed = 1e3 / u.micron
dnu_fixed = 1e-3 / u.micron

# --- vary dnu ---
dnu_list = (10.0 ** np.linspace(-5, -2, 16)) / u.micron
eps_dnu = []
for dnu in dnu_list:
    val = integrate_function(B_of_nutilde, nu_min_fixed, nu_max_fixed, dnu, method="trapz")
    eps_dnu.append(precision(val, truth))

plt.figure(figsize=(7.5, 5.5))
plt.plot(dnu_list.value, eps_dnu, linewidth=2.5)
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$d\tilde{\nu}\ (\mu\mathrm{m}^{-1})$")
plt.ylabel(r"Precision $\left|1-\mathrm{calc}/\mathrm{truth}\right|$")
plt.tight_layout()
plt.savefig("precision_vs_dnu.svg", format="svg")
plt.close()

# --- vary nu_max ---
nu_max_list = (10.0 ** np.linspace(0, 4, 17)) / u.micron
eps_numax = []
for nu_max in nu_max_list:
    val = integrate_function(B_of_nutilde, nu_min_fixed, nu_max, dnu_fixed, method="trapz")
    eps_numax.append(precision(val, truth))

plt.figure(figsize=(7.5, 5.5))
plt.plot(nu_max_list.value, eps_numax, linewidth=2.5)
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$\tilde{\nu}_{\max}\ (\mu\mathrm{m}^{-1})$")
plt.ylabel(r"Precision $\left|1-\mathrm{calc}/\mathrm{truth}\right|$")
plt.tight_layout()
plt.savefig("precision_vs_numax.svg", format="svg")
plt.close()

# --- vary nu_min ---
nu_min_list = (10.0 ** np.linspace(-10, -2, 17)) / u.micron
eps_numin = []
for nu_min in nu_min_list:
    val = integrate_function(B_of_nutilde, nu_min, nu_max_fixed, dnu_fixed, method="trapz")
    eps_numin.append(precision(val, truth))

plt.figure(figsize=(7.5, 5.5))
plt.plot(nu_min_list.value, eps_numin, linewidth=2.5)
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$\tilde{\nu}_{\min}\ (\mu\mathrm{m}^{-1})$")
plt.ylabel(r"Precision $\left|1-\mathrm{calc}/\mathrm{truth}\right|$")
plt.tight_layout()
plt.savefig("precision_vs_numin.svg", format="svg")
plt.close()

#(c) Bias
bias = 1e-12 * truth.unit

eps_dnu_biased = []
for dnu in dnu_list:
    val = integrate_function(B_of_nutilde, nu_min_fixed, nu_max_fixed, dnu, method="trapz")
    val_b = val + bias
    eps_dnu_biased.append(precision(val_b, truth))

plt.figure(figsize=(7.5, 5.5))
plt.plot(dnu_list.value, eps_dnu, linewidth=2.5, label="No bias")
plt.plot(dnu_list.value, eps_dnu_biased, linewidth=2.5, label="Biased")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$d\tilde{\nu}\ (\mu\mathrm{m}^{-1})$")
plt.ylabel(r"Precision $\left|1-\mathrm{calc}/\mathrm{truth}\right|$")
plt.legend()
plt.tight_layout()
plt.savefig("precision_vs_dnu_bias.svg", format="svg")
plt.close()

#(d) Convergence test
def rel_change(a, b):
    return abs(((b - a) / b).decompose().value)

dnu_list = (10.0 ** np.linspace(-5, -2, 16)) / u.micron
conv_est = []

for dnu in dnu_list:
    I1 = integrate_function(B_of_nutilde, nu_min_fixed, nu_max_fixed, dnu, method="trapz")
    I2 = integrate_function(B_of_nutilde, nu_min_fixed, nu_max_fixed, dnu/2, method="trapz")
    conv_est.append(rel_change(I1, I2))

plt.figure(figsize=(7.5, 5.5))
plt.plot(dnu_list.value, conv_est, linewidth=2.5)
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$d\tilde{\nu}\ (\mu\mathrm{m}^{-1})$")
plt.ylabel(r"Convergence estimate $\left|I(d/2)-I(d)\right|/I(d/2)$")
plt.tight_layout()
plt.savefig("convergence_test_dnu.svg", format="svg")
plt.close()