import pandas as pd
import numpy as np
from astro530.partition import saha_phi, partition_function
from astro530.pe_solver import load_solar_abundances, solve_pe_gray_9p8, available_species_for_pe
from astro530.opacity import kappa_hminus_bf, kappa_hminus_ff, kappa_h_bf, kappa_h_ff
from astropy.constants import sigma_T
import astropy.units as u

def load_valiiic(path):
    cols = [
        "h_km", "m_gcm2", "tau500", "T_K", "V_kms",
        "n_H_cm3", "n_e_cm3", "Ptotal_dyncm2",
        "Pgas_over_Ptotal", "rho_gcm3"
    ]

    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=cols,
        engine="python"
    )

    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Pgas_dyncm2"] = df["Pgas_over_Ptotal"] * df["Ptotal_dyncm2"]
    df["Pe_from_neT"] = df["n_e_cm3"] * 1.380649e-16 * df["T_K"]   # k_B in cgs
    df["logtau500"] = np.log10(df["tau500"].replace(0, np.nan))

    return df

def hydrogen_lte_ionization_from_valiiic(df, saha_phi):
    out = df.copy()

    np_list = []
    n0_list = []
    frac_list = []

    for _, row in out.iterrows():
        T = row["T_K"]
        Pe = row["Pe_from_neT"]
        nH = row["n_H_cm3"]

        # H -> H+ ; use u(H+) = 1, u(H) = 2 approximately
        phi_H = saha_phi(T, 13.6, u_upper=1.0, u_lower=2.0)
        r = phi_H / Pe

        np_H = (r / (1.0 + r)) * nH
        n0_H = nH / (1.0 + r)
        frac = np_H / nH

        np_list.append(np_H)
        n0_list.append(n0_H)
        frac_list.append(frac)

    out["n_p_LTE"] = np.array(np_list)
    out["n_H0_LTE"] = np.array(n0_list)
    out["x_HII_LTE"] = np.array(frac_list)

    return out

def first_ion_fraction(symbol, T, Pe, ioniz, part_table, partition_function, saha_phi):
    I1 = ioniz[symbol]["chis_eV"][0]
    u0, _ = partition_function(symbol, T, part_table)
    u1, _ = partition_function(symbol + "+", T, part_table)
    phi = saha_phi(T, I1, u_upper=u1, u_lower=u0)
    r = phi / Pe
    f1 = r / (1.0 + r)
    f0 = 1.0 / (1.0 + r)
    return f0, f1

def abundance_lookup(abund_df):
    return abund_df.set_index("element")["A"].to_dict()

def lte_electron_contributions(T, Pe, abund_df, ioniz, part_table,
                               partition_function, saha_phi,
                               species_list=None):
    if species_list is None:
        species_list = available_species_for_pe(ioniz, part_table, abund_df, partition_function)

    A = abund_df.set_index("element")["A"].to_dict()

    numerators = {}
    total = 0.0

    for sym in species_list:
        f0, f1 = first_ion_fraction(sym, T, Pe, ioniz, part_table, partition_function, saha_phi)
        contrib = A[sym] * f1
        numerators[sym] = contrib
        total += contrib

    fractions = {sym: numerators[sym] / total for sym in numerators}
    return fractions, total


def valiiic_lte_pe_and_fractions(val_df, abund_df, ioniz, part_table,
                                 partition_function, saha_phi, solve_pe_gray_9p8):
    out = val_df.copy()

    pe_lte = []

    h0, h1 = [], []
    fe0, fe1 = [], []
    mg0, mg1 = [], []
    si0, si1 = [], []

    h_contrib = []
    fe_contrib = []
    mg_contrib = []
    si_contrib = []

    species_list = available_species_for_pe(ioniz, part_table, abund_df, partition_function)

    for _, row in out.iterrows():
        T = row["T_K"]
        Pg = row["Pgas_dyncm2"]

        Pe, hist = solve_pe_gray_9p8(
            T=T,
            Pg=Pg,
            abund_df=abund_df,
            ioniz=ioniz,
            part_table=part_table,
            partition_function=partition_function,
            verbose=False
        )

        pe_lte.append(Pe)

        # ionization fractions
        f0, f1 = first_ion_fraction("H", T, Pe, ioniz, part_table, partition_function, saha_phi)
        h0.append(f0); h1.append(f1)

        f0, f1 = first_ion_fraction("Fe", T, Pe, ioniz, part_table, partition_function, saha_phi)
        fe0.append(f0); fe1.append(f1)

        f0, f1 = first_ion_fraction("Mg", T, Pe, ioniz, part_table, partition_function, saha_phi)
        mg0.append(f0); mg1.append(f1)

        f0, f1 = first_ion_fraction("Si", T, Pe, ioniz, part_table, partition_function, saha_phi)
        si0.append(f0); si1.append(f1)

        # contribution fractions to LTE ne
        fracs, total = lte_electron_contributions(
            T, Pe, abund_df, ioniz, part_table,
            partition_function, saha_phi,
            species_list=species_list
        )

        h_contrib.append(fracs.get("H", 0.0))
        fe_contrib.append(fracs.get("Fe", 0.0))
        mg_contrib.append(fracs.get("Mg", 0.0))
        si_contrib.append(fracs.get("Si", 0.0))

    out["Pe_LTE"] = np.array(pe_lte)

    out["HI_frac"] = np.array(h0)
    out["HII_frac"] = np.array(h1)

    out["FeI_frac"] = np.array(fe0)
    out["FeII_frac"] = np.array(fe1)

    out["MgI_frac"] = np.array(mg0)
    out["MgII_frac"] = np.array(mg1)

    out["SiI_frac"] = np.array(si0)
    out["SiII_frac"] = np.array(si1)

    out["H_contrib_ne"] = np.array(h_contrib)
    out["Fe_contrib_ne"] = np.array(fe_contrib)
    out["Mg_contrib_ne"] = np.array(mg_contrib)
    out["Si_contrib_ne"] = np.array(si_contrib)

    return out

k_B_cgs = 1.380649e-16
amu_g = 1.66053906660e-24


def stimulated_emission_factor(lam_a, T):
    lam_a = np.asarray(lam_a, dtype=float)
    theta = 5040.0 / T
    chi_lambda = 1.2398e4 / lam_a
    return 1.0 - 10.0**(-chi_lambda * theta)


def grams_per_H_particle(abund_df):
    df = abund_df.copy()
    return float((df["A"] * df["weight"] * amu_g).sum())


def hydrogen_neutral_fraction(T, Pe, ioniz, part_table, partition_function, saha_phi):
    f0, f1 = first_ion_fraction("H", T, Pe, ioniz, part_table, partition_function, saha_phi)
    return f0


def electron_donors_per_H(T, Pe, abund_df, ioniz, part_table, partition_function, saha_phi):
    """
    electrons per H nucleus in the single-ionization approximation:
        sum_j A_j f_j,II
    """
    A = abund_df.set_index("element")["A"].to_dict()
    species = available_species_for_pe(ioniz, part_table, abund_df, partition_function)

    total = 0.0
    for sym in species:
        _, f1 = first_ion_fraction(sym, T, Pe, ioniz, part_table, partition_function, saha_phi)
        total += A[sym] * f1

    return total


def kappa_electron_scattering(T, Pe, abund_df, ioniz, part_table, partition_function, saha_phi):
    """
    Thomson scattering opacity in cm^2/g.
    """
    ne_per_H = electron_donors_per_H(T, Pe, abund_df, ioniz, part_table, partition_function, saha_phi)
    gram_per_H = grams_per_H_particle(abund_df)
    sigmaT = sigma_T.cgs.value
    return sigmaT * ne_per_H / gram_per_H

def kappa_continuum_total(lam_a, T, Pg, Pe, abund_df, ioniz, part_table,
                          partition_function, saha_phi):
    """
    Total continuum opacity in cm^2/g.
    """
    stim = stimulated_emission_factor(lam_a, T)
    xH0 = hydrogen_neutral_fraction(T, Pe, ioniz, part_table, partition_function, saha_phi)
    gram_per_H = grams_per_H_particle(abund_df)

    # per neutral H atom pieces from Problem 13
    k_hm_bf = kappa_hminus_bf(lam_a, T, Pe)
    k_hm_ff = kappa_hminus_ff(lam_a, T, Pe)
    k_h_bf_ = kappa_h_bf(lam_a, T)
    k_h_ff_ = kappa_h_ff(lam_a, T)

    # convert to cm^2/g
    # Hbf, Hff, H- bf get stimulated emission factor
    # H- ff already has it built into Gray's fit
    kap_hm_bf = k_hm_bf * stim * xH0 / gram_per_H
    kap_hm_ff = k_hm_ff * xH0 / gram_per_H
    kap_h_bf  = k_h_bf_ * stim * xH0 / gram_per_H
    kap_h_ff  = k_h_ff_ * stim * xH0 / gram_per_H

    kap_e = kappa_electron_scattering(T, Pe, abund_df, ioniz, part_table,
                                      partition_function, saha_phi)

    kap_total = kap_hm_bf + kap_hm_ff + kap_h_bf + kap_h_ff + kap_e

    return {
        "stim": stim,
        "xH0": xH0,
        "gram_per_H": gram_per_H,
        "kappa_total": kap_total,
        "kappa_Hminus_bf": kap_hm_bf,
        "kappa_Hminus_ff": kap_hm_ff,
        "kappa_H_bf": kap_h_bf,
        "kappa_H_ff": kap_h_ff,
        "kappa_e": kap_e,
    }

def infer_kappa500_hse(val_df):
    g = 10.0**4.4377
    kappa_bins = []
    tau_mid = []
    h_mid = []

    for i in range(len(val_df) - 1):
        dP = val_df.iloc[i+1]["Ptotal_dyncm2"] - val_df.iloc[i]["Ptotal_dyncm2"]
        dtau = val_df.iloc[i+1]["tau500"] - val_df.iloc[i]["tau500"]

        if dP == 0:
            kappa = np.nan
        else:
            kappa = g * dtau / dP

        kappa_bins.append(kappa)
        tau_mid.append(0.5 * (val_df.iloc[i+1]["tau500"] + val_df.iloc[i]["tau500"]))
        h_mid.append(0.5 * (val_df.iloc[i+1]["h_km"] + val_df.iloc[i]["h_km"]))

    return pd.DataFrame({
        "h_mid_km": h_mid,
        "tau500_mid": tau_mid,
        "kappa500_hse": kappa_bins
    })