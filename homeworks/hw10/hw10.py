import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from astro530.valiii import load_valiiic, kappa_continuum_total
from astro530.partition import saha_phi, partition_function, load_partition_table, load_ioniz
from astro530.pe_solver import load_solar_abundances, solve_pe_gray_9p8
from astro530.broadening import sigma_naD_lambda_single, kappa_line_nad, sigma_line_nu, sigma_naD_lambda, na_lines

c = 2.99792458e10          # cm/s
def sigma_single_line_lambda(lambda_A, line, T=5770, xi_kms=1.0, ne=1e13, Pgas=1e5):
    lambda_cm = np.asarray(lambda_A) * 1e-8
    nu = c / lambda_cm

    sigma, info = sigma_line_nu(
        nu,
        lambda0_A=line["lambda0_A"],
        f=line["f"],
        A_ul=line["A_ul"],
        T=T,
        xi_kms=xi_kms,
        ne=ne,
        Pgas=Pgas
    )
    return sigma, info


lam = np.linspace(5887, 5899, 4000)

sigma_D2, info_D2 = sigma_single_line_lambda(lam, na_lines["D2"], T=5770, xi_kms=1.0, ne=1e13, Pgas=1e5)
sigma_D1, info_D1 = sigma_single_line_lambda(lam, na_lines["D1"], T=5770, xi_kms=1.0, ne=1e13, Pgas=1e5)
sigma_tot = sigma_D2 + sigma_D1

print("D2 broadening terms:")
for k, v in info_D2.items():
    print(f"{k}: {v:.3e}")

print("\nD1 broadening terms:")
for k, v in info_D1.items():
    print(f"{k}: {v:.3e}")

print("Max sigma_D2 =", np.max(sigma_D2))
print("Max sigma_D1 =", np.max(sigma_D1))
print("Max sigma_total =", np.max(sigma_tot))

plt.figure(figsize=(10,5))
plt.plot(lam, sigma_D2, label='Na I D2 (5889.95 Å)', alpha=0.8)
plt.plot(lam, sigma_D1, label='Na I D1 (5895.92 Å)', alpha=0.8)
plt.plot(lam, sigma_tot, color='k', lw=2, label='Total')

plt.xlabel(r'Wavelength [$\AA$]')
plt.ylabel(r'$\sigma_\nu^\ell$')
plt.title('Na I D Doublet Monochromatic Line Extinction per Particle')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig("naD_sigma_lambda.pdf")

val = load_valiiic("../hw9/VALIIIC_sci_e.txt")

idx_tau1 = np.argmin(np.abs(val["tau500"] - 1.0))
row = val.iloc[idx_tau1]

print("Using VALIIIC row:")
print(row)

T = row["T_K"]
Pg = row["Pgas_dyncm2"]
rho = row["rho_gcm3"]

abund_df = load_solar_abundances("../hw8/SolarAbundance.txt")

part_table = load_partition_table("../hw6/RepairedPartitionFunctions.txt")
ioniz = load_ioniz("../hw6/ioniz.txt")

# LTE electron pressure at this depth
Pe, hist = solve_pe_gray_9p8(
    T=T,
    Pg=Pg,
    abund_df=abund_df,
    ioniz=ioniz,
    part_table=part_table,
    partition_function=partition_function,
    verbose=False
)

print(f"T = {T:.1f} K")
print(f"Pg = {Pg:.3e} dyn/cm^2")
print(f"rho = {rho:.3e} g/cm^3")
print(f"Pe = {Pe:.3e} dyn/cm^2")

# -----------------------------
# Wavelength grid
# -----------------------------
lam = np.linspace(5888.0, 5898.0, 4000)

# -----------------------------
# Line opacities
# -----------------------------
kap_D2 = kappa_line_nad(
    lam, T, rho, Pe, abund_df, ioniz, part_table,
    partition_function, saha_phi,
    line_name="D2"
)

kap_D1 = kappa_line_nad(
    lam, T, rho, Pe, abund_df, ioniz, part_table,
    partition_function, saha_phi,
    line_name="D1"
)

# -----------------------------
# Continuum opacities
# -----------------------------
kap_cont = kappa_continuum_total(
    lam, T, Pg, Pe, abund_df, ioniz, part_table,
    partition_function, saha_phi
)

# -----------------------------
# Print atomic properties
# -----------------------------
print("\nAtomic properties:")
print(f"Ground fraction  = {kap_D2['ground_fraction']:.4f}")
print(f"Neutral fraction = {kap_D2['neutral_fraction']:.4e}")
print(f"Stimulated emission factor (5890A) = {kap_D2['stim'][np.argmin(np.abs(lam-5890))]:.4f}")
print(f"Stimulated emission factor (5896A) = {kap_D1['stim'][np.argmin(np.abs(lam-5896))]:.4f}")

# -----------------------------
# Print opacity table values
# -----------------------------
for wl in [5890.0, 5893.0, 5896.0]:
    i = np.argmin(np.abs(lam - wl))
    print(f"\nAt {wl:.0f} A:")
    print(f"kappa(NaD1)      = {kap_D1['kappa_line'][i]:.4g}")
    print(f"kappa(NaD2)      = {kap_D2['kappa_line'][i]:.4g}")
    print(f"kappa(continuum) = {kap_cont['kappa_total'][i]:.4g}")
    print(f"kappa(H- bf)     = {kap_cont['kappa_Hminus_bf'][i]:.4g}")
    print(f"kappa(H- ff)     = {kap_cont['kappa_Hminus_ff'][i]:.4g}")
    print(f"kappa(H bf)      = {kap_cont['kappa_H_bf'][i]:.4g}")
    print(f"kappa(H ff)      = {kap_cont['kappa_H_ff'][i]:.4g}")
    print(f"kappa(e-)        = {kap_cont['kappa_e']:.4g}")
    # exact line-center values
i_d2 = np.argmin(np.abs(lam - na_lines["D2"]["lambda0_A"]))
i_d1 = np.argmin(np.abs(lam - na_lines["D1"]["lambda0_A"]))

print("\nExact-center opacities:")
print(f"D2 center ({lam[i_d2]:.3f} A): {kap_D2['kappa_line'][i_d2]:.4g}")
print(f"D1 center ({lam[i_d1]:.3f} A): {kap_D1['kappa_line'][i_d1]:.4g}")

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10,6))
plt.plot(lam, kap_D2["kappa_line"], label="Na D2")
plt.plot(lam, kap_D1["kappa_line"], label="Na D1")
plt.plot(lam, kap_cont["kappa_Hminus_bf"], label=r"H$^-$ bf")
plt.plot(lam, kap_cont["kappa_Hminus_ff"], label=r"H$^-$ ff")
plt.plot(lam, kap_cont["kappa_H_bf"], label="H bf")
plt.plot(lam, kap_cont["kappa_H_ff"], label="H ff")
plt.plot(lam, np.full_like(lam, kap_cont["kappa_e"]), label=r"e$^-$")
plt.plot(lam, kap_cont["kappa_total"], color="k", lw=2, label="continuum total")

plt.yscale("log")
plt.xlabel(r"Wavelength [$\AA$]")
plt.ylabel(r"$\kappa_\nu$ [cm$^2$/g]")
plt.title(r"Na I D Line and Continuum Opacity at $\tau_{500}\approx1$")
plt.legend(fontsize=9, ncol=2)
plt.tight_layout()
plt.savefig("na_d_opacities.pdf")