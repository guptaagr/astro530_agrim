# hw1.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from astropy import units as u

from astro530.planck import planck_wavenumber

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.formatter.useoffset": False,
})

nu_tilde_lin = np.linspace(1e-3, 12.0, 2000) / u.micron # avoid zero to prevent issues with log scale
nu_tilde_log = np.logspace(-1.0, 1.2, 2000) / u.micron

temps = [10000, 7000, 3000] * u.K
unit_Bnu = u.W / (u.m**2 * u.Hz * u.sr)

B_lin = {T.value: planck_wavenumber(nu_tilde_lin, T).to(unit_Bnu) for T in temps}
for T in temps:
    Bvals = B_lin[T.value].value
    nuvals = nu_tilde_lin.value
    peak_idx = np.argmax(Bvals)
    peak_nu = nuvals[peak_idx]
    print(f"T = {int(T.value)} K: peak wavenumber = {peak_nu:.3f} micron^-1")

B_log = {T.value: planck_wavenumber(nu_tilde_log, T).to(unit_Bnu) for T in temps}

# Plot 1: B_nu vs wavenumber
plt.figure(figsize=(7.5, 5.5))
ax = plt.gca()
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.ticklabel_format(axis="both", style="plain", useOffset=False)
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
ax = plt.gca()
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.ticklabel_format(axis="both", style="plain", useOffset=False)
for T in temps:
    plt.plot(nu_tilde_log.value, B_log[T.value].value, linewidth=2,
             label=rf"$T={int(T.value)}\ \mathrm{{K}}$")
plt.xscale("log")
plt.xlabel(r"Wavenumber $\tilde{\nu}\ (\mu\mathrm{m}^{-1})$")
plt.ylabel(r"$B_\nu\ \left[\mathrm{W\ m^{-2}\ Hz^{-1}\ sr^{-1}}\right]$")
plt.xlim(10**(-1.0), 10**(1.2))
plt.legend()
plt.tight_layout()
plt.savefig("planck_Bnu_vs_lognutilde.svg")
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