import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.special import expn
from scipy.integrate import cumulative_trapezoid as cumtrapz

from astro530.valiii import load_valiiic, infer_kappa500_hse
from astro530.partition import saha_phi, partition_function, load_partition_table, load_ioniz
from astro530.pe_solver import load_solar_abundances
from astro530.opacity import (
    kappa_hminus_bf, kappa_hminus_ff, kappa_h_bf, kappa_h_ff,
    chi_lambda_ev, theta_5040
)
from astro530.broadening import sigma_naD_lambda_single, na_lines

# -----------------------------
# constants
# -----------------------------
c = 2.99792458e10       # cm/s
h = 6.62607015e-27      # erg s
k_B = 1.380649e-16      # erg/K
sigmaT_cgs = 6.6524587321e-25
amu_g = 1.66053906660e-24


# -----------------------------
# helpers
# -----------------------------
def planck_nu(nu, T):
    x = h * nu / (k_B * T)
    return (2.0 * h * nu**3 / c**2) / np.expm1(x)


def H_nu_surface(tau_nu, S_nu):
    E2 = expn(2, tau_nu)
    return 0.5 * np.trapezoid(S_nu * E2, tau_nu)


def grams_per_H_particle(abund_df):
    return float((abund_df["A"] * abund_df["weight"] * amu_g).sum())


def log_interp1d(x_new, x, y, floor=1e-99):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_new = np.asarray(x_new, dtype=float)

    good = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x = x[good]
    y = y[good]

    x_new_safe = np.clip(x_new, np.min(x), None)
    y_safe = np.clip(y, floor, None)

    return 10**np.interp(np.log10(x_new_safe), np.log10(x), np.log10(y_safe))


def stimulated_emission_factor(lam_a, T):
    lam_a = np.asarray(lam_a, dtype=float)
    return 1.0 - 10.0**(-chi_lambda_ev(lam_a) * theta_5040(T))


def first_ion_fraction(symbol, T, Pe, ioniz, part_table):
    I1 = ioniz[symbol]["chis_eV"][0]
    u0, _ = partition_function(symbol, T, part_table)
    u1, _ = partition_function(symbol + "+", T, part_table)
    phi = saha_phi(T, I1, u_upper=u1, u_lower=u0)
    r = phi / Pe
    f0 = 1.0 / (1.0 + r)
    f1 = r / (1.0 + r)
    return f0, f1


def hydrogen_neutral_fraction(T, Pe, ioniz, part_table):
    f0, _ = first_ion_fraction("H", T, Pe, ioniz, part_table)
    return f0


def sodium_neutral_and_ground_fraction(T, Pe, ioniz, part_table):
    # Na I / Na total
    f0, _ = first_ion_fraction("Na", T, Pe, ioniz, part_table)

    # ground fraction within neutral Na
    u0, _ = partition_function("Na", T, part_table)
    ground_fraction = 2.0 / u0   # ground state g=2, E=0

    return f0, ground_fraction


def electron_donors_per_H(T, Pe, abund_df, ioniz, part_table):
    A = abund_df.set_index("element")["A"].to_dict()
    total = 0.0

    for sym in A:
        if sym in ioniz and sym in part_table and (sym + "+") in part_table:
            try:
                _, f1 = first_ion_fraction(sym, T, Pe, ioniz, part_table)
                total += A[sym] * f1
            except Exception:
                pass

    return total


def kappa_electron_scattering_from_pe_rho(T, Pe, rho):
    ne = Pe / (k_B * T)
    return sigmaT_cgs * ne / rho


def continuum_opacity_cm2g(lam_a, T, Pe, rho, abund_df, ioniz, part_table):
    """
    Total continuum opacity in cm^2/g using VALIIIC Pe and rho.
    """
    xH0 = hydrogen_neutral_fraction(T, Pe, ioniz, part_table)
    gram_per_H = grams_per_H_particle(abund_df)
    stim = stimulated_emission_factor(lam_a, T)

    k_hm_bf = kappa_hminus_bf(lam_a, T, Pe)
    k_hm_ff = kappa_hminus_ff(lam_a, T, Pe)
    k_h_bf_ = kappa_h_bf(lam_a, T)
    k_h_ff_ = kappa_h_ff(lam_a, T)

    kap_hm_bf = k_hm_bf * stim * xH0 / gram_per_H
    kap_hm_ff = k_hm_ff * xH0 / gram_per_H
    kap_h_bf  = k_h_bf_ * stim * xH0 / gram_per_H
    kap_h_ff  = k_h_ff_ * stim * xH0 / gram_per_H
    kap_e     = kappa_electron_scattering_from_pe_rho(T, Pe, rho)

    return {
        "kappa_total": kap_hm_bf + kap_hm_ff + kap_h_bf + kap_h_ff + kap_e,
        "kappa_Hminus_bf": kap_hm_bf,
        "kappa_Hminus_ff": kap_hm_ff,
        "kappa_H_bf": kap_h_bf,
        "kappa_H_ff": kap_h_ff,
        "kappa_e": kap_e,
    }


def line_opacity_cm2g(lam_a, T, Pe, Pg, abund_df, ioniz, part_table, line_name):
    """
    Na D line opacity in cm^2/g using VALIIIC Pe.
    """
    A_Na = float(abund_df.loc[abund_df["element"] == "Na", "A"].values[0])
    gram_per_H = grams_per_H_particle(abund_df)

    neutral_fraction, ground_fraction = sodium_neutral_and_ground_fraction(
        T, Pe, ioniz, part_table
    )

    n_lower_per_H = A_Na * neutral_fraction * ground_fraction
    n_lower_per_g = n_lower_per_H / gram_per_H

    ne = Pe / (k_B * T)

    sigma_line, _ = sigma_naD_lambda_single(
        lam_a,
        line_name=line_name,
        T=T,
        xi_kms=1.0,
        ne=ne,
        Pgas=Pg
    )

    stim = stimulated_emission_factor(lam_a, T)
    kappa_line = n_lower_per_g * sigma_line * stim

    return kappa_line


# -----------------------------
# load atmosphere + tables
# -----------------------------
val = load_valiiic("../hw9/VALIIIC_sci_e.txt")
kap500_df = infer_kappa500_hse(val)

abund_df = load_solar_abundances("../hw8/SolarAbundance.txt")
part_table = load_partition_table("../hw6/RepairedPartitionFunctions.txt")
ioniz = load_ioniz("../hw6/ioniz.txt")

# attach kappa500 to original VAL grid using interpolation in tau500
tau_val = val["tau500"].values
kap500_mid_tau = kap500_df["tau500_mid"].values
kap500_mid_val = kap500_df["kappa500_hse"].values
kap500_on_val = log_interp1d(tau_val, kap500_mid_tau, kap500_mid_val)

val = val.copy()
val["kappa500_cm2g"] = kap500_on_val

# use only positive tau500 points and sort increasing
mask = np.isfinite(val["tau500"]) & (val["tau500"] > 0)
atm = val.loc[mask].sort_values("tau500").reset_index(drop=True)

tau500_orig = atm["tau500"].values
T_orig = atm["T_K"].values
Pg_orig = atm["Pgas_dyncm2"].values
Pe_orig = atm["Pe_from_neT"].values   # use VALIIIC ne directly
kap500_orig = atm["kappa500_cm2g"].values

# fine tau500 grid
tau500_fine = np.logspace(np.log10(tau500_orig.min()), np.log10(tau500_orig.max()), 600)
T_fine = log_interp1d(tau500_fine, tau500_orig, T_orig)
Pg_fine = log_interp1d(tau500_fine, tau500_orig, Pg_orig)
Pe_fine = log_interp1d(tau500_fine, tau500_orig, Pe_orig)
kap500_fine = log_interp1d(tau500_fine, tau500_orig, kap500_orig)

# wavelength grid
lam_wide = np.linspace(5888.0, 5898.0, 1200)
lam_d2 = np.linspace(5889.6, 5890.3, 1200)
lam_d1 = np.linspace(5895.6, 5896.3, 1200)
lam = np.unique(np.sort(np.concatenate([lam_wide, lam_d2, lam_d1])))
nu = c / (lam * 1e-8)

Fnu = np.zeros_like(lam)
Fnu_cont = np.zeros_like(lam)

# store tau_nu map for plotting
tau_nu_map = np.zeros((len(lam), len(tau500_fine)))
tau_nu_cont_map = np.zeros((len(lam), len(tau500_fine)))

rho_orig = atm["rho_gcm3"].values
rho_fine = log_interp1d(tau500_fine, tau500_orig, rho_orig)

for j, lam_j in enumerate(lam):
    # continuum over all depths
    kap_cont_depth = np.zeros(len(tau500_fine))
    kap_d2_depth = np.zeros(len(tau500_fine))
    kap_d1_depth = np.zeros(len(tau500_fine))
    if j == 0:
        tmp = continuum_opacity_cm2g(lam_j, T_fine[0], Pe_fine[0], rho_fine[0], abund_df, ioniz, part_table)
        print(tmp)

    for i in range(len(tau500_fine)):
        kap_cont_depth[i] = continuum_opacity_cm2g(
            lam_j, T_fine[i], Pe_fine[i], rho_fine[i], abund_df, ioniz, part_table
        )["kappa_total"]

        kap_d2_depth[i] = line_opacity_cm2g(
            lam_j, T_fine[i], Pe_fine[i], Pg_fine[i],
            abund_df, ioniz, part_table, "D2"
        )

        kap_d1_depth[i] = line_opacity_cm2g(
            lam_j, T_fine[i], Pe_fine[i], Pg_fine[i],
            abund_df, ioniz, part_table, "D1"
        )

    kap_tot_depth = kap_cont_depth + kap_d1_depth + kap_d2_depth

    ratio_tot = kap_tot_depth / kap500_fine
    ratio_cont = kap_cont_depth / kap500_fine

    tau_nu = np.concatenate([[0.0], cumtrapz(ratio_tot, tau500_fine)])
    tau_nu_cont = np.concatenate([[0.0], cumtrapz(ratio_cont, tau500_fine)])

    tau_nu_map[j, :] = tau_nu
    tau_nu_cont_map[j, :] = tau_nu_cont

    nu_j = c / (lam_j * 1e-8)
    Snu = planck_nu(nu_j, T_fine)

    Hnu = H_nu_surface(tau_nu, Snu)
    Hnu_cont = H_nu_surface(tau_nu_cont, Snu)

    Fnu[j] = 4.0 * np.pi * Hnu
    Fnu_cont[j] = 4.0 * np.pi * Hnu_cont

    if j % 25 == 0:
        print(f"{j+1}/{len(lam)} wavelengths done")

print(np.nanmin(kap_cont_depth), np.nanmax(kap_cont_depth))
print(np.nanmin(kap_d2_depth), np.nanmax(kap_d2_depth))
print(np.nanmin(kap_d1_depth), np.nanmax(kap_d1_depth))
print(np.nanmin(kap500_fine), np.nanmax(kap500_fine))
print(np.all(np.isfinite(ratio_tot)))
print(np.all(np.isfinite(tau_nu)))
print(np.all(np.isfinite(Snu)))

# convert to F_lambda
lam_cm = lam * 1e-8
Flam = Fnu * c / lam_cm**2 * 1e-8        # per Angstrom
Flam_cont = Fnu_cont * c / lam_cm**2 * 1e-8

# -----------------------------
# plots
# -----------------------------
plt.figure(figsize=(8, 5))
for target_lam in [5889.95, 5893.0, 5895.92]:
    idx = np.argmin(np.abs(lam - target_lam))
    plt.loglog(tau500_fine, tau_nu_map[idx, :], label=fr"$\lambda={lam[idx]:.2f}\,\AA$")
plt.plot(tau500_fine, tau500_fine, "k--", lw=1, label=r"$\tau_\nu=\tau_{500}$")
plt.xlabel(r"$\tau_{500}$")
plt.ylabel(r"$\tau_\nu$")
plt.title(r"Optical depth mapping near Na I D")
plt.legend()
plt.tight_layout()
plt.savefig("tau_nu_map.pdf")

plt.figure(figsize=(9, 5))
plt.plot(lam, Flam_cont, label="continuum")
plt.plot(lam, Flam, label="line + continuum")
plt.xlabel(r"Wavelength [$\AA$]")
plt.ylabel(r"$F_\lambda(0)$ [cgs / $\AA$]")
plt.title(r"Emergent LTE Flux near Na I D")
plt.legend()
plt.tight_layout()
plt.savefig("na_d_flux.pdf")

plt.figure(figsize=(9, 5))
plt.plot(lam, Flam / Flam_cont)
plt.axhline(1.0, color="k", ls="--", lw=1)
plt.xlabel(r"Wavelength [$\AA$]")
plt.ylabel(r"$F_\lambda / F_{\lambda,\mathrm{cont}}$")
plt.title(r"Normalized LTE Na I D Profile")
plt.tight_layout()
plt.savefig("na_d_profile.pdf")

plt.figure(figsize=(8, 5))
pcm = plt.pcolormesh(
    lam, tau500_fine, tau_nu_map.T,
    shading="auto",
    norm=LogNorm(vmin=1e-6, vmax=np.nanmax(tau_nu_map)),
    rasterized=True
)
plt.yscale("log")
plt.colorbar(pcm, label=r"$\tau_\nu$")
plt.xlabel(r"Wavelength [$\AA$]")
plt.ylabel(r"$\tau_{500}$")
plt.title(r"$\tau_\nu(\tau_{500}, \lambda)$")
plt.tight_layout()
plt.savefig("tau_nu_map_color.pdf", bbox_inches="tight")