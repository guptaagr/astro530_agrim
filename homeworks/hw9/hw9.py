import matplotlib.pyplot as plt
import numpy as np
from astro530.valiii import load_valiiic, hydrogen_lte_ionization_from_valiiic, valiiic_lte_pe_and_fractions, kappa_continuum_total, infer_kappa500_hse
from astro530.partition import saha_phi, partition_function, load_ioniz, load_partition_table
from astro530.pe_solver import load_solar_abundances, solve_pe_gray_9p8

val = load_valiiic("VALIIIC_sci_e.txt")
val_b = hydrogen_lte_ionization_from_valiiic(val, saha_phi)

# Problem 15
plt.figure()
plt.plot(val_b["h_km"], np.log10(val_b["n_p_LTE"]), label="LTE $n_p$")
plt.plot(val_b["h_km"], np.log10(val_b["n_e_cm3"]), label="VALIII C $n_e$")
plt.xlabel("h (km)")
plt.ylabel(r"$\log_{10} n\ \mathrm{(cm^{-3})}$")
plt.xlim(0,800)
plt.legend()
plt.gca().invert_xaxis()
plt.savefig("valiii.pdf")

def hydrogen_np_from_T(Tion, Pe, nH, saha_phi):
    phi_H = saha_phi(Tion, 13.6, u_upper=1.0, u_lower=2.0)
    r = phi_H / Pe
    return (r / (1.0 + r)) * nH

def find_ionization_temperature(np_target, Pe, nH, saha_phi,
                                Tmin=3000.0, Tmax=20000.0, ngrid=20000):
    Ts = np.linspace(Tmin, Tmax, ngrid)
    nps = np.array([hydrogen_np_from_T(T, Pe, nH, saha_phi) for T in Ts])
    idx = np.argmin(np.abs(nps - np_target))
    return Ts[idx], nps[idx]

row800 = val.iloc[np.argmin(np.abs(val["h_km"] - 800))]
Pe800 = row800["Pe_from_neT"]
nH800 = row800["n_H_cm3"]

np_target = 1e11

Tion, np_check = find_ionization_temperature(np_target, Pe800, nH800, saha_phi)
print(Tion, np_check)

ioniz = load_ioniz("../hw6/ioniz.txt")
part = load_partition_table("../hw6/RepairedPartitionFunctions.txt")
abund = load_solar_abundances("../hw8/SolarAbundance.txt")

val = load_valiiic("VALIIIC_sci_e.txt")
val_c = valiiic_lte_pe_and_fractions(
    val, abund, ioniz, part,
    partition_function, saha_phi, solve_pe_gray_9p8
)

plt.figure()
plt.plot(val_c["h_km"], val_c["H_contrib_ne"], label="H")
plt.plot(val_c["h_km"], val_c["Fe_contrib_ne"], label="Fe")
plt.plot(val_c["h_km"], val_c["Mg_contrib_ne"], label="Mg")
plt.plot(val_c["h_km"], val_c["Si_contrib_ne"], label="Si")
plt.xlabel("h (km)")
plt.ylabel(r"contributions to $n_e$")
plt.xlim(800, 0)
plt.ylim(0.1, 1.0)
plt.legend()
plt.tight_layout()
plt.savefig("valiii_contrib.pdf")

plt.figure()
plt.plot(val_c["h_km"], val_c["HII_frac"], label="H II")
plt.plot(val_c["h_km"], val_c["FeII_frac"], label="Fe II")
plt.plot(val_c["h_km"], val_c["MgII_frac"], label="Mg II")
plt.plot(val_c["h_km"], val_c["SiII_frac"], label="Si II")
plt.xlabel("h (km)")
plt.ylabel("Ionized fraction")
plt.xlim(800, 0)
plt.legend()
plt.tight_layout()
plt.savefig("valiii_ionfrac.pdf")

# print(val_c['H_contrib_ne']+val_c['Fe_contrib_ne']+val_c['Mg_contrib_ne']+val_c['Si_contrib_ne'])

# Problem 16
plt.figure()
plt.plot(val_c["h_km"], np.log10(val_c["Pe_LTE"]), label=r"LTE $P_e$")
plt.plot(val_c["h_km"], np.log10(val_c["Pe_from_neT"]), label=r"VALIII C $P_e=n_e kT$")
plt.xlabel("h (km)")
plt.ylabel(r"$\log_{10} P_e\ \mathrm{(dyne\ cm^{-2})}$")
plt.xlim(800, 0)
plt.legend()
plt.tight_layout()
plt.savefig("valiiic_pe_compare.pdf")

cases = [
    {"T": 6420.0, "Pe": 57.0, "Pg": 1.13e5, "lam": 5000.0},
    {"T": 11572.0, "Pe": 10.0**2.76, "Pg": 1259.0, "lam": 15000.0},
]

for case in cases:
    out = kappa_continuum_total(
        lam_a=case["lam"],
        T=case["T"],
        Pg=case["Pg"],
        Pe=case["Pe"],
        abund_df=abund,
        ioniz=ioniz,
        part_table=part,
        partition_function=partition_function,
        saha_phi=saha_phi,
    )

    print("\nCase:", case)
    print("stim =", out["stim"])
    print("xH0 =", out["xH0"])
    print("sum Aj mu_j (g) =", out["gram_per_H"])
    print("kappa(total) =", out["kappa_total"])
    print("kappa(H-_bf) =", out["kappa_Hminus_bf"])
    print("kappa(H-_ff) =", out["kappa_Hminus_ff"])
    print("kappa(H_bf)  =", out["kappa_H_bf"])
    print("kappa(H_ff)  =", out["kappa_H_ff"])
    print("kappa(e-)    =", out["kappa_e"])

kap500 = []
for _, row in val_c.iterrows():
    out = kappa_continuum_total(
        lam_a=5000.0,
        T=row["T_K"],
        Pg=row["Pgas_dyncm2"],
        Pe=row["Pe_LTE"],
        abund_df=abund,
        ioniz=ioniz,
        part_table=part,
        partition_function=partition_function,
        saha_phi=saha_phi,
    )
    kap500.append(out["kappa_total"])

val_c["kappa500_LTE"] = np.array(kap500)

plt.figure()
plt.plot(np.log10(val_c["tau500"]), np.log10(val_c["kappa500_LTE"]))
plt.xlabel(r"$\log_{10}\tau_{500}$")
plt.ylabel(r"$\log_{10}\kappa_{500}\ (\mathrm{cm^2\,g^{-1}})$")
plt.tight_layout()
plt.savefig("kappa500_vs_tau.pdf")

hse_df = infer_kappa500_hse(val)

plt.figure()
plt.plot(np.log10(val_c["tau500"]), np.log10(val_c["kappa500_LTE"]), label="Opacity calculator")
plt.plot(np.log10(hse_df["tau500_mid"]), np.log10(hse_df["kappa500_hse"]), label="Hydrostatic inference")
plt.xlabel(r"$\log_{10}\tau_{500}$")
plt.ylabel(r"$\log_{10}\kappa_{500}\ (\mathrm{cm^2\,g^{-1}})$")
plt.legend()
plt.tight_layout()
plt.savefig("kappa500_compare.pdf")
