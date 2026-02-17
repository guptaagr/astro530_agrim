import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from scipy.special import expn

from astro530.integration import trapz
from astro530.planck import planck_wavenumber 

def H_nu_surface(tau, S_func):
    tau = u.Quantity(tau).to(u.dimensionless_unscaled)
    S = S_func(tau)
    E2 = expn(2, tau.value)
    integrand = S * E2
    return 0.5 * trapz(tau, integrand)    # H_nu(0)

def F_nu_surface(tau, S_func):
    return 4 * np.pi * H_nu_surface(tau, S_func)

Teff = 8700 * u.K

def T_of_tau(tau):
    tau = u.Quantity(tau).to(u.dimensionless_unscaled)
    return Teff * ( (3/4) * (tau + 2/3) )**(1/4)

tau_min = 1e-6 * u.dimensionless_unscaled
tau_max = 100.0 * u.dimensionless_unscaled
N_tau = 2000
tau_grid = np.logspace(np.log10(tau_min.value), np.log10(tau_max.value), N_tau) * u.dimensionless_unscaled

lam_min = 0.05 * u.micron     # avoid exactly 0
lam_max = 12.0 * u.micron
N_lam = 400
lam_grid = np.logspace(np.log10(lam_min.value), np.log10(lam_max.value), N_lam) * u.micron
nutilde_grid = (1 / lam_grid).to(1 / u.micron)   # wavenumber in 1/micron

Fnu_vals = []

# (b)
Fnu_exact = []
for nutilde in nutilde_grid:
    def S_grey(tau, nutilde=nutilde):
        return planck_wavenumber(nutilde, T_of_tau(tau))  # intensity per sr
    Fnu_exact.append(F_nu_surface(tau_grid, S_grey))

Fnu_exact = u.Quantity(Fnu_exact)

# (c)
tau_EB = (2/3) * u.dimensionless_unscaled
T_EB = T_of_tau(tau_EB)

Fnu_EB = np.pi * planck_wavenumber(nutilde_grid, T_EB)  # intensity per sr -> multiply by π gives flux

plt.figure(figsize=(7.5, 5.5))
plt.plot(nutilde_grid.value, Fnu_exact.value, lw=2.5, label="Numerical (E2 integral)")
plt.plot(nutilde_grid.value, Fnu_EB.value, lw=2.5, ls="--", label=r"Eddington--Barbier: $\pi S_\nu(2/3)$")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$\tilde{\nu}\ (\mu{\rm m}^{-1})$")
plt.ylabel(r"$F_\nu(0)$ (arbitrary units)")
plt.legend()
plt.tight_layout()
plt.savefig("p8d_Fnu_exact_vs_EB.pdf")
plt.close()

# Fractional difference plot
frac = ((Fnu_EB - Fnu_exact) / Fnu_exact).decompose().value

plt.figure(figsize=(7.5, 5.5))
plt.plot(nutilde_grid.value, frac, lw=2.5)
plt.xscale("log")
plt.axhline(0, lw=1)  # nice visual reference
plt.xlabel(r"$\tilde{\nu}\ (\mu{\rm m}^{-1})$")
plt.ylabel(r"Signed fractional difference $(F_{\nu,\mathrm{EB}}-F_{\nu})/F_{\nu}$")
plt.tight_layout()
plt.savefig("p8e_signed_fracdiff.pdf")
plt.close()
