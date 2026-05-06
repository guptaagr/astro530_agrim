import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expn
from scipy.integrate import cumulative_trapezoid
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter

from astro530.valiii import load_valiiic, infer_kappa500_hse
from astro530.partition import saha_phi, partition_function, load_partition_table, load_ioniz
from astro530.pe_solver import load_solar_abundances
from astro530.opacity import (
    kappa_hminus_bf, kappa_hminus_ff, kappa_h_bf, kappa_h_ff,
    chi_lambda_ev, theta_5040
)
from astro530.broadening import sigma_naD_lambda_single


# ============================================================
# Paths
# ============================================================

VAL_PATH = "../hw9/VALIIIC_sci_e.txt"
ABUND_PATH = "../hw8/SolarAbundance.txt"
PARTITION_PATH = "../hw6/RepairedPartitionFunctions.txt"
IONIZ_PATH = "../hw6/ioniz.txt"

SOURCE_PATH = "CoreS2014.txt"
B3S_PATH = "3sdep_extrap.txt"


# ============================================================
# Constants
# ============================================================

c = 2.99792458e10
h = 6.62607015e-27
k_B = 1.380649e-16
sigmaT_cgs = 6.6524587321e-25
amu_g = 1.66053906660e-24


# ============================================================
# Helper functions
# ============================================================

def planck_nu(nu, T):
    x = h * nu / (k_B * T)
    return (2.0 * h * nu**3 / c**2) / np.expm1(x)

def planck_lambda_per_nm(lam_a, T):
    lam_cm = np.asarray(lam_a, dtype=float) * 1e-8
    x = h * c / (lam_cm * k_B * T)

    # B_lambda per cm
    B_lam_per_cm = (2.0 * h * c**2 / lam_cm**5) / np.expm1(x)

    # convert per cm -> per nm
    return B_lam_per_cm * 1e-7


def formal_flux(tau_nu_depth_lam, S_depth_lam):
    """
    F_nu^+(0) = 2 pi int S_nu E2(tau_nu) d tau_nu
    depth axis = 0, wavelength axis = 1
    """
    E2 = expn(2, tau_nu_depth_lam)
    integrand = S_depth_lam * E2
    return 2.0 * np.pi * np.trapezoid(integrand, tau_nu_depth_lam, axis=0)


def grams_per_H_particle(abund_df):
    return float((abund_df["A"] * abund_df["weight"] * amu_g).sum())


def interp_logx_linear_y(x_new, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_new = np.asarray(x_new, dtype=float)

    good = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x = x[good]
    y = y[good]

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_new_safe = np.clip(x_new, x.min(), x.max())
    return np.interp(np.log10(x_new_safe), np.log10(x), y)


def interp_logx_logy(x_new, x, y, floor=1e-99):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_new = np.asarray(x_new, dtype=float)

    good = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x = x[good]
    y = y[good]

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_new_safe = np.clip(x_new, x.min(), x.max())
    y_safe = np.clip(y, floor, None)

    return 10.0**np.interp(np.log10(x_new_safe), np.log10(x), np.log10(y_safe))


def interp_height_linear(h_new, h, y):
    h = np.asarray(h, dtype=float)
    y = np.asarray(y, dtype=float)
    h_new = np.asarray(h_new, dtype=float)

    good = np.isfinite(h) & np.isfinite(y)
    h = h[good]
    y = y[good]

    order = np.argsort(h)
    h = h[order]
    y = y[order]

    h_new_safe = np.clip(h_new, h.min(), h.max())
    return np.interp(h_new_safe, h, y)


def stimulated_emission_factor(lam_a, T):
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


def sodium_lte_ground_fraction(T, Pe, ioniz, part_table):
    """
    LTE fraction of all Na atoms in the Na I 3s lower level.
    """
    f_NaI, _ = first_ion_fraction("Na", T, Pe, ioniz, part_table)

    u_NaI, _ = partition_function("Na", T, part_table)
    ground_frac_within_NaI = 2.0 / u_NaI

    return f_NaI * ground_frac_within_NaI


def continuum_opacity_depth_lambda(lam_a, T, Pe, rho, abund_df, ioniz, part_table):
    """
    Continuum opacity in cm^2/g at one depth over all wavelengths.
    Uses VALIII ne through Pe, and VALIII rho for electron scattering.
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
    kap_h_bf = k_h_bf_ * stim * xH0 / gram_per_H
    kap_h_ff = k_h_ff_ * stim * xH0 / gram_per_H

    ne = Pe / (k_B * T)
    kap_e = sigmaT_cgs * ne / rho

    return kap_hm_bf + kap_hm_ff + kap_h_bf + kap_h_ff + kap_e


def line_opacity_lte_depth_lambda(lam_a, T, Pe, Pg, abund_df, ioniz, part_table, line_name):
    """
    LTE Na D line opacity in cm^2/g at one depth over all wavelengths.
    """
    A_Na = float(abund_df.loc[abund_df["element"] == "Na", "A"].values[0])
    gram_per_H = grams_per_H_particle(abund_df)

    lower_frac_lte = sodium_lte_ground_fraction(T, Pe, ioniz, part_table)
    n_lower_per_H_lte = A_Na * lower_frac_lte
    n_lower_per_g_lte = n_lower_per_H_lte / gram_per_H

    ne = Pe / (k_B * T)

    sigma_line, _ = sigma_naD_lambda_single(
        lam_a,
        line_name=line_name,
        T=T,
        xi_kms=1.0,
        ne=ne,
        Pgas=Pg,
    )

    stim = stimulated_emission_factor(lam_a, T)

    return n_lower_per_g_lte * sigma_line * stim


# ============================================================
# Load atmosphere and tables
# ============================================================

val = load_valiiic(VAL_PATH)
kap500_df = infer_kappa500_hse(val)

abund_df = load_solar_abundances(ABUND_PATH)
part_table = load_partition_table(PARTITION_PATH)
ioniz = load_ioniz(IONIZ_PATH)

# attach kappa_500 inferred from hydrostatic equilibrium
tau_val = val["tau500"].values
kap500_on_val = interp_logx_logy(
    tau_val,
    kap500_df["tau500_mid"].values,
    kap500_df["kappa500_hse"].values,
)

val = val.copy()
val["kappa500_cm2g"] = kap500_on_val

atm = val[np.isfinite(val["tau500"]) & (val["tau500"] > 0)].copy()
atm = atm.sort_values("tau500").reset_index(drop=True)

tau500_orig = atm["tau500"].values
h_orig = atm["h_km"].values
T_orig = atm["T_K"].values
Pg_orig = atm["Pgas_dyncm2"].values
Pe_orig = atm["Pe_from_neT"].values
rho_orig = atm["rho_gcm3"].values
kap500_orig = atm["kappa500_cm2g"].values

# fine depth grid
tau500_fine = np.logspace(np.log10(tau500_orig.min()), np.log10(tau500_orig.max()), 700)

h_fine = interp_logx_linear_y(tau500_fine, tau500_orig, h_orig)
T_fine = interp_logx_logy(tau500_fine, tau500_orig, T_orig)
Pg_fine = interp_logx_logy(tau500_fine, tau500_orig, Pg_orig)
Pe_fine = interp_logx_logy(tau500_fine, tau500_orig, Pe_orig)
rho_fine = interp_logx_logy(tau500_fine, tau500_orig, rho_orig)
kap500_fine = interp_logx_logy(tau500_fine, tau500_orig, kap500_orig)


# ============================================================
# Load NLTE source function and departure coefficient
# ============================================================

coreS = np.loadtxt(SOURCE_PATH)
dep3s = np.loadtxt(B3S_PATH)

h_coreS = coreS[:, 0]
S_coreS = coreS[:, 1]

h_dep = dep3s[:, 0]
b3s_dep = dep3s[:, 1]

S_nlte_fine = interp_height_linear(h_fine, h_coreS, S_coreS)
# ------------------------------------------------------------
# Anchor NLTE source function to Planck function in LTE region
# ------------------------------------------------------------

Blam_anchor = planck_lambda_per_nm(5895.924, T_fine)

# pick a deep layer (closest to h ~ 0 km)
i_anchor = np.argmin(np.abs(h_fine - 0.0))

scale_factor = Blam_anchor[i_anchor] / S_nlte_fine[i_anchor]
S_nlte_fine *= scale_factor
b3s_fine = interp_height_linear(h_fine, h_dep, b3s_dep)


# ============================================================
# Wavelength grid
# ============================================================

lam_wide = np.linspace(5888.0, 5898.0, 1000)
lam_d2 = np.linspace(5889.65, 5890.25, 1200)
lam_d1 = np.linspace(5895.65, 5896.25, 1200)
lam = np.unique(np.sort(np.concatenate([lam_wide, lam_d2, lam_d1])))

nu = c / (lam * 1e-8)

Ndepth = len(tau500_fine)
Nlam = len(lam)


# ============================================================
# Build opacity matrices
# ============================================================

kap_cont = np.zeros((Ndepth, Nlam))
kap_d2_lte = np.zeros((Ndepth, Nlam))
kap_d1_lte = np.zeros((Ndepth, Nlam))

print("Building opacity matrices...")

for i in range(Ndepth):
    kap_cont[i, :] = continuum_opacity_depth_lambda(
        lam, T_fine[i], Pe_fine[i], rho_fine[i],
        abund_df, ioniz, part_table
    )

    kap_d2_lte[i, :] = line_opacity_lte_depth_lambda(
        lam, T_fine[i], Pe_fine[i], Pg_fine[i],
        abund_df, ioniz, part_table, "D2"
    )

    kap_d1_lte[i, :] = line_opacity_lte_depth_lambda(
        lam, T_fine[i], Pe_fine[i], Pg_fine[i],
        abund_df, ioniz, part_table, "D1"
    )

    if i % 100 == 0:
        print(f"  depth {i+1}/{Ndepth}")

# NLTE opacity: n_l^NLTE = b_3s n_l^LTE
kap_d2_nlte = b3s_fine[:, None] * kap_d2_lte
kap_d1_nlte = b3s_fine[:, None] * kap_d1_lte

kap_line_lte = kap_d1_lte + kap_d2_lte
kap_line_nlte = kap_d1_nlte + kap_d2_nlte

kap_tot_lte = kap_cont + kap_line_lte
kap_tot_nlte = kap_cont + kap_line_nlte


# ============================================================
# Optical depths
# ============================================================

ratio_cont = kap_cont / kap500_fine[:, None]
ratio_lte = kap_tot_lte / kap500_fine[:, None]
ratio_nlte = kap_tot_nlte / kap500_fine[:, None]

tau_cont = np.vstack([
    np.zeros(Nlam),
    cumulative_trapezoid(ratio_cont, tau500_fine, axis=0)
])

tau_lte = np.vstack([
    np.zeros(Nlam),
    cumulative_trapezoid(ratio_lte, tau500_fine, axis=0)
])

tau_nlte = np.vstack([
    np.zeros(Nlam),
    cumulative_trapezoid(ratio_nlte, tau500_fine, axis=0)
])


# ============================================================
# Source functions and fluxes
# ============================================================

# Work in wavelength units because CoreS2014.txt is in source-function
# units per nm, matching Rutten Fig. 10.13.
Blam = planck_lambda_per_nm(lam[None, :], T_fine[:, None])

# LTE source function
S_lte = Blam

# NLTE approximation: use supplied line-core source function,
# assumed frequency/wavelength independent over the narrow Na D interval.
S_nlte = S_nlte_fine[:, None] * np.ones((Ndepth, Nlam))

print("Computing formal solution...")

Flam_cont_lte = formal_flux(tau_cont, S_lte)
Flam_lte = formal_flux(tau_lte, S_lte)

Flam_cont_nlte = formal_flux(tau_cont, S_nlte)
Flam_nlte = formal_flux(tau_nlte, S_nlte)

norm_lte = Flam_lte / Flam_cont_lte
norm_nlte = Flam_nlte / Flam_cont_nlte

# For plotting only
Blam_plot = planck_lambda_per_nm(5895.924, T_fine)

# ============================================================
# Diagnostics
# ============================================================

eta_nlte = kap_line_nlte / kap_cont

bad_region = (tau_nlte > 0.1) & (eta_nlte < 10.0) & (h_fine[:, None] > 300.0)

print("\nDiagnostics:")
print(f"max b_3s = {np.nanmax(b3s_fine):.3g}")
print(f"min b_3s = {np.nanmin(b3s_fine):.3g}")
print(f"max eta_NLTE = {np.nanmax(eta_nlte):.3e}")
print(f"Any region with tau_nu > 0.1, eta < 10, h > 300 km? {np.any(bad_region)}")

for wl in [5889.95, 5893.0, 5895.92]:
    j = np.argmin(np.abs(lam - wl))
    idx1 = np.where(tau_nlte[:, j] >= 1.0)[0]
    if len(idx1) > 0:
        i1 = idx1[0]
        print(f"tau_NLTE = 1 at lambda={lam[j]:.3f} A: h ~ {h_fine[i1]:.1f} km, T ~ {T_fine[i1]:.0f} K")
    else:
        print(f"tau_NLTE never reaches 1 at lambda={lam[j]:.3f} A")


# ============================================================
# Plots
# ============================================================

# 1. Input source function and B_nu comparison
plt.figure(figsize=(7, 5))
plt.semilogx(S_nlte_fine, h_fine, label=r"input $S_\lambda^{\rm tot}$")
plt.semilogx(Blam_plot, h_fine, label=r"$B_\lambda(T)$ at D$_1$")
plt.xlabel(r"Source function [erg cm$^{-2}$ s$^{-1}$ nm$^{-1}$ sr$^{-1}$]")
plt.ylabel(r"Height [km]")
plt.title(r"NLTE Source Function Approximation")
plt.legend()
plt.tight_layout()
plt.savefig("source_function_nlte.pdf", bbox_inches="tight")
plt.show()

# 2. Departure coefficient
plt.figure(figsize=(7, 5))
plt.semilogx(b3s_fine, h_fine)
plt.xlabel(r"$b_{3s}$")
plt.ylabel(r"Height [km]")
plt.title(r"Na I $3s$ Departure Coefficient")
plt.tight_layout()
plt.savefig("b3s_departure.pdf", bbox_inches="tight")
plt.show()

# 3. Tau map
plt.figure(figsize=(8, 5))
pcm = plt.pcolormesh(
    lam, tau500_fine, tau_nlte,
    shading="auto",
    norm=LogNorm(vmin=1e-6, vmax=np.nanmax(tau_nlte)),
    rasterized=True,
)
plt.yscale("log")
plt.colorbar(pcm, label=r"$\tau_\nu$")
plt.xlabel(r"Wavelength [$\AA$]")
plt.ylabel(r"$\tau_{500}$")
plt.title(r"NLTE $\tau_\nu(\tau_{500},\lambda)$")
plt.tight_layout()
plt.savefig("tau_nlte_map.pdf", bbox_inches="tight")
plt.show()

# 4. LTE vs NLTE normalized profiles
plt.figure(figsize=(9, 5))
plt.plot(lam, norm_lte, label="LTE", alpha=0.8)
plt.plot(lam, norm_nlte, label="NLTE approximation", lw=2)
plt.axhline(1.0, color="k", ls="--", lw=1)
plt.xlabel(r"Wavelength [$\AA$]")
plt.ylabel(r"$F_\nu/F_{\nu,\mathrm{cont}}$")
plt.title(r"Na I D: LTE vs NLTE Approximation")
plt.legend()
plt.tight_layout()
plt.savefig("naD_lte_vs_nlte.pdf", bbox_inches="tight")
plt.show()

# 5. Absolute NLTE flux
ax = plt.gca()
ax.ticklabel_format(style='plain', axis='y')
Flam_cont_nlte_scaled = Flam_cont_nlte / 1e7
Flam_nlte_scaled = Flam_nlte / 1e7

plt.figure(figsize=(9, 5))
plt.plot(lam, Flam_cont_nlte_scaled, label="continuum")
plt.plot(lam, Flam_nlte_scaled, label="line + continuum")
plt.xlabel(r"Wavelength [$\AA$]")
plt.ylabel(r"$F_\lambda$ [$10^7$ erg cm$^{-2}$ s$^{-1}$ nm$^{-1}$]")
plt.title(r"Approximate NLTE Emergent Flux near Na I D")
plt.legend()
plt.tight_layout()
plt.savefig("naD_nlte_flux.pdf", bbox_inches="tight")
plt.show()