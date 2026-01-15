import numpy as np
from astropy import units as u
from astropy.constants import h, c, k_B
from astropy.modeling.models import BlackBody

def planck_nu(nu, T):
    """
    Planck function B_nu(T): specific intensity per unit frequency.

    Inputs:
        nu : frequency-like Quantity
        T  : temperature Quantity (K)

    Returns:
        B_nu with units W m^-2 Hz^-1 sr^-1
    """
    nu = u.Quantity(nu).to(u.Hz, equivalencies=u.spectral())
    T = u.Quantity(T).to(u.K)

    x = (h * nu / (k_B * T)).decompose().value
    pref = (2 * h * nu**3 / c**2).to(u.W / (u.m**2 * u.Hz))
    Bnu = (pref / np.expm1(x)) / u.sr
    return Bnu.to(u.W / (u.m**2 * u.Hz * u.sr))


def planck_lambda(lam, T):
    """
    Planck function B_lambda(T): specific intensity per unit wavelength.

    Inputs:
        lam : wavelength-like Quantity
        T   : temperature Quantity (K)

    Returns:
        B_lambda with units W m^-3 sr^-1
    """
    lam = u.Quantity(lam).to(u.m, equivalencies=u.spectral())
    T = u.Quantity(T).to(u.K)

    x = (h * c / (lam * k_B * T)).decompose().value
    pref = (2 * h * c**2 / lam**5).to(u.W / (u.m**3))
    Blam = (pref / np.expm1(x)) / u.sr
    return Blam.to(u.W / (u.m**3 * u.sr))


def planck_wavenumber(nu_tilde, T):
    """
    B_nu(T) evaluated as a function of wavenumber (1 / lambda).

    Inputs:
        nu_tilde : wavenumber Quantity (e.g. 1 / micron)
        T        : temperature Quantity (K)

    Returns:
        B_nu(T) in W m^-2 Hz^-1 sr^-1
    """
    nu_tilde = u.Quantity(nu_tilde).to(1 / u.m, equivalencies=u.spectral())
    lam = (1 / nu_tilde).to(u.m)
    nu = (c / lam).to(u.Hz)
    return planck_nu(nu, T)

# Example usage and comparison with Astropy's BlackBody model
T = 7000 * u.K
nu = np.array([1e12, 2e12, 3e12]) * u.Hz

bb = BlackBody(temperature=T)

B_mine = planck_nu(nu, T)

print(bb / B_mine)