import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz

c = 2.99792458e10          # cm/s
k_B = 1.380649e-16         # erg/K
m_e = 9.10938356e-28       # g
m_p = 1.6726219e-24        # g
e_charge = 4.80320425e-10  # statcoulomb
h = 6.62607015e-27         # erg s
amu = 1.66053906660e-24    # g
a0 = 5.29177210903e-9      # cm (Bohr radius)
ryd_eV = 13.605693         # eV

na_lines = {
    "D2": {
        "lambda0_A": 5890,   # Angstrom
        "f": 0.641,
        "g_l": 2,
        "g_u": 4,
        "A_ul": 6.14e7,          # s^-1, approximate
        "E_lower_eV": 0.0,       # ground state
        "E_upper_eV": 2.104
    },
    "D1": {
        "lambda0_A": 5896,   # Angstrom
        "f": 0.320,
        "g_l": 2,
        "g_u": 2,
        "A_ul": 6.14e7,          # s^-1, approximate
        "E_lower_eV": 0.0,
        "E_upper_eV": 2.102
    }
}

def grams_per_H_particle(abund_df):
    df = abund_df.copy()
    return float((df["A"] * df["weight"] * amu).sum())

def doppler_width_nu(lambda0_cm, T=5770, xi_kms=1.0, atomic_mass_amu=22.989769):
    """
    Doppler width in frequency units (Hz).
    Includes thermal + microturbulent broadening.
    """
    nu0 = c / lambda0_cm
    m_atom = atomic_mass_amu * amu
    xi = xi_kms * 1e5  # km/s -> cm/s

    vD = np.sqrt(2 * k_B * T / m_atom + xi**2)
    delta_nu_D = (nu0 / c) * vD
    return delta_nu_D, vD

def gamma_rad(A_ul):
    """Natural/radiative damping constant in s^-1."""
    return A_ul

def gamma_stark(ne=1e13, T=5770):
    """
    Very approximate quadratic Stark broadening for Na I D.
    Returns s^-1.

    For solar photosphere Na D, this is usually much smaller than vdW.
    You can refine this if your class notes give a preferred formula.
    """
    # crude scaling
    return 1e-8 * ne

def effective_n_squared(chi_level_eV, Z=1):
    """
    n_*^2 = R Z^2 / (E_inf - E_n)
    where chi_level_eV = ionization energy from that level.
    """
    return ryd_eV * Z**2 / chi_level_eV


def mean_square_radius(nstar2, l, Z=1):
    """
    Bates-Damgaard approximation for <r^2> in atomic units.
    """
    return (nstar2 / (2 * Z**2)) * (5 * nstar2 + 1 - 3 * l * (l + 1))


def gamma_vdw_unsold(T=5770, Pgas=1e5):
    """
    Unsöld / Gray approximate vdW broadening for Na D.
    Returns gamma_6 in s^-1.

    Pgas should be in dyn/cm^2.
    """
    # Na I ionization energy from ground
    chi_ion = 5.139  # eV

    # lower = 3s
    chi_lower = chi_ion - 0.0
    n2_lower = effective_n_squared(chi_lower, Z=1)
    r2_lower = mean_square_radius(n2_lower, l=0, Z=1)

    # upper = 3p
    E_upper = 2.104  # eV
    chi_upper = chi_ion - E_upper
    n2_upper = effective_n_squared(chi_upper, Z=1)
    r2_upper = mean_square_radius(n2_upper, l=1, Z=1)

    delta_r2 = abs(r2_upper - r2_lower)

    log_gamma6 = 6.33 + 0.4*np.log10(delta_r2) + np.log10(Pgas) - 0.7*np.log10(T)
    return 10**log_gamma6

def voigt_H(a, u):
    """Dimensionless Voigt H(a,u) using scipy's Faddeeva."""
    return np.real(wofz(u + 1j*a))


def phi_nu(nu, nu0, delta_nu_D, gamma_total):
    """
    Normalized Voigt profile in frequency units:
    integral phi_nu dnu = 1
    """
    u = (nu - nu0) / delta_nu_D
    a = gamma_total / (4 * np.pi * delta_nu_D)
    H = voigt_H(a, u)
    return H / (np.sqrt(np.pi) * delta_nu_D)

def sigma_line_nu(nu, lambda0_A, f, A_ul, T=5770, xi_kms=1.0,
                  ne=1e13, Pgas=1e5, atomic_mass_amu=22.989769):
    """
    Monochromatic line extinction per particle sigma_nu^l (cm^2 / Hz? no:
    cm^2 when phi_nu is per Hz and prefactor has cm^2 Hz).
    """
    lambda0_cm = lambda0_A * 1e-8
    nu0 = c / lambda0_cm

    delta_nu_D, vD = doppler_width_nu(lambda0_cm, T=T, xi_kms=xi_kms,
                                      atomic_mass_amu=atomic_mass_amu)

    g_rad = gamma_rad(A_ul)
    g_stark = gamma_stark(ne=ne, T=T)
    g_vdw = gamma_vdw_unsold(T=T, Pgas=Pgas)

    gamma_tot = g_rad + g_stark + g_vdw

    phi = phi_nu(nu, nu0, delta_nu_D, gamma_tot)

    sigma = (np.pi * e_charge**2 / (m_e * c)) * f * phi

    return sigma, {
        "nu0": nu0,
        "delta_nu_D": delta_nu_D,
        "vD_cm_s": vD,
        "gamma_rad": g_rad,
        "gamma_stark": g_stark,
        "gamma_vdw": g_vdw,
        "gamma_total": gamma_tot
    }

def sigma_naD_lambda_single(lambda_A, line_name="D2", T=5770, xi_kms=1.0, ne=1e13, Pgas=1e5):
    """
    Single Na I D line monochromatic extinction per particle vs wavelength.
    """
    lambda_cm = np.asarray(lambda_A) * 1e-8
    nu = c / lambda_cm

    line = na_lines[line_name]

    sigma, details = sigma_line_nu(
        nu,
        lambda0_A=line["lambda0_A"],
        f=line["f"],
        A_ul=line["A_ul"],
        T=T,
        xi_kms=xi_kms,
        ne=ne,
        Pgas=Pgas if Pgas is not None else 1e5
    )

    return sigma, details

def sigma_naD_lambda(lambda_A, T=5770, xi_kms=1.0, ne=1e13, Pgas=1e5):
    """
    Total Na I D monochromatic extinction per particle as a function of wavelength.
    lambda_A can be array.
    """
    lambda_cm = np.asarray(lambda_A) * 1e-8
    nu = c / lambda_cm

    sigma_tot = np.zeros_like(lambda_cm)

    details = {}

    for name, line in na_lines.items():
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
        sigma_tot += sigma
        details[name] = info

    return sigma_tot, details

def stim_emission_line(lam_a, T):
    lam_a = np.asarray(lam_a, dtype=float)
    theta = 5040.0 / T
    chi_lambda = 1.2398e4 / lam_a
    return 1.0 - 10.0**(-chi_lambda * theta)

def sodium_neutral_and_ground_fraction(T, Pe, ioniz, part_table, partition_function, saha_phi):
    """
    Returns:
        neutral_fraction = Na I / total Na
        ground_fraction  = population of 3s level / Na I
    """
    I1 = ioniz["Na"]["chis_eV"][0]

    u0, levels0 = partition_function("Na", T, part_table)
    u1, levels1 = partition_function("Na+", T, part_table)

    phi = saha_phi(T, I1, u_upper=u1, u_lower=u0)
    r = phi / Pe

    neutral_fraction = 1.0 / (1.0 + r)
    ground_fraction = 2.0 / u0

    return neutral_fraction, ground_fraction

def kappa_line_nad(lam_a, T, rho, Pe, abund_df, ioniz, part_table,
                   partition_function, saha_phi,
                   line_name="D2"):
    """
    Na I D line opacity in cm^2/g
    """

    lam_a = np.asarray(lam_a, dtype=float)

    # sodium abundance relative to H
    A_Na = float(abund_df.loc[abund_df["element"] == "Na", "A"].values[0])

    # number of grams per H nucleus
    gram_per_H = grams_per_H_particle(abund_df)

    # Na neutral fraction and ground-state fraction
    neutral_fraction, ground_fraction = sodium_neutral_and_ground_fraction(
        T, Pe, ioniz, part_table, partition_function, saha_phi
    )

    # number of absorbers in lower level per H nucleus
    n_lower_per_H = A_Na * neutral_fraction * ground_fraction

    # convert to absorbers per gram
    n_lower_per_g = n_lower_per_H / gram_per_H

    # line profile from Problem 20
    sigma_line, _ = sigma_naD_lambda_single(
        lam_a,
        line_name=line_name,
        T=T,
        xi_kms=1.0,
        ne=Pe / (1.380649e-16 * T),
        Pgas=None
    )

    stim = stim_emission_line(lam_a, T)
    kappa_line = n_lower_per_g * sigma_line * stim

    return {
        "kappa_line": kappa_line,
        "neutral_fraction": neutral_fraction,
        "ground_fraction": ground_fraction,
        "stim": stim,
        "n_lower_per_g": n_lower_per_g
    }