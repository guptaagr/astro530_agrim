# hw1.py
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from radiation import planck_wavenumber

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "legend.fontsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.formatter.useoffset": False,
})

nu_tilde_lin = np.linspace(1e-3, 12.0, 2000) / u.micron
nu_tilde_log = np.logspace(-1.0, 1.2, 2000) / u.micron

temps = [10000, 7000, 3000] * u.K
unit_Bnu = u.W / (u.m**2 * u.Hz * u.sr)

B_lin = {T.value: planck_wavenumber(nu_tilde_lin, T).to(unit_Bnu) for T in temps}
B_log = {T.value: planck_wavenumber(nu_tilde_log, T).to(unit_Bnu) for T in temps}

# Plot 1: B_nu vs wavenumber
plt.figure(figsize=(7.5, 5.5))
for T in temps:
    plt.plot(nu_tilde_lin.value, B_lin[T.value].value, linewidth=2,
             label=rf"$T={int(T.value)}\ \mathrm{{K}}$")
plt.xlabel(r"Wavenumber $\tilde{\nu}\ (\mu\mathrm{m}^{-1})$")
plt.ylabel(r"$B_\nu\ \left[\mathrm{W\ m^{-2}\ Hz^{-1}\ sr^{-1}}\right]$")
plt.xlim(0, 12)
plt.ylim(0, 1.05 * max(B_lin[10000.0].value))
plt.legend()
plt.tight_layout()
plt.savefig("planck_Bnu_vs_nutilde.svg")
plt.close()

# Plot 2: log10(B_nu) vs wavenumber
plt.figure(figsize=(7.5, 5.5))
for T in temps:
    plt.plot(nu_tilde_lin.value, np.log10(B_lin[T.value].value), linewidth=2,
             label=rf"$T={int(T.value)}\ \mathrm{{K}}$")
plt.xlabel(r"Wavenumber $\tilde{\nu}\ (\mu\mathrm{m}^{-1})$")
plt.ylabel(r"$\log_{10}\, B_\nu\ \left[\mathrm{W\ m^{-2}\ Hz^{-1}\ sr^{-1}}\right]$")
plt.xlim(0, 12)
plt.legend()
plt.tight_layout()
plt.savefig("planck_logBnu_vs_nutilde.svg")
plt.close()

# Plot 3: log-log (log tick marks)
plt.figure(figsize=(7.5, 5.5))
for T in temps:
    plt.plot(nu_tilde_log.value, B_log[T.value].value, linewidth=2,
             label=rf"$T={int(T.value)}\ \mathrm{{K}}$")
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"Wavenumber $\tilde{\nu}\ (\mu\mathrm{m}^{-1})$")
plt.ylabel(r"$B_\nu\ \left[\mathrm{W\ m^{-2}\ Hz^{-1}\ sr^{-1}}\right]$")
plt.xlim(10**(-1.0), 10**(1.2))
plt.legend()
plt.tight_layout()
plt.savefig("planck_Bnu_vs_nutilde_loglog.svg")
plt.close()