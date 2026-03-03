import numpy as np

lam_lyman_a = 912
i_h_ev = 13.6
loge10 = np.log10(np.e)

def theta_5040(t):
    t = np.asarray(t, dtype=float)
    return 5040.0 / t

def chi_lambda_ev(lam_a):
    lam_a = np.asarray(lam_a, dtype=float)
    return 1.2398e4 / lam_a

def chi_exc_h_ev(n):
    n = np.asarray(n, dtype=float)
    return i_h_ev * (1.0 - 1.0 / (n**2))

def alpha0_hydrogen_cgs():
    return 1.0449e-26

def lamr_unitless(lam_a):
    lam_a = np.asarray(lam_a, dtype=float)
    return lam_a / lam_lyman_a

def n0_from_lambda(lam_a):
    lam_a = np.asarray(lam_a, dtype=float)
    n0 = np.ceil(np.sqrt(lam_a / lam_lyman_a)).astype(int)
    return np.maximum(n0, 1)

def g_bf(lam_a, n):
    lam_a = np.asarray(lam_a, dtype=float)
    n = np.asarray(n, dtype=float)
    lamr = lamr_unitless(lam_a)
    return 1.0 - (0.3456 / (lamr ** (1.0/3.0))) * ((lamr / (n**2)) - 0.5)

def g_ff(lam_a, t):
    lam_a = np.asarray(lam_a, dtype=float)
    t = np.asarray(t, dtype=float)
    th = theta_5040(t)
    lamr = lamr_unitless(lam_a)
    chi_lam = chi_lambda_ev(lam_a)
    return 1.0 + (0.3456 / (lamr ** (1.0/3.0))) * (loge10 / (th * chi_lam) + 0.5)

def kappa_h_ff(lam_a, t, i_ev=i_h_ev):
    lam_a = np.asarray(lam_a, dtype=float)
    t = np.asarray(t, dtype=float)
    a0 = alpha0_hydrogen_cgs()
    th = theta_5040(t)
    return (a0 * (lam_a**3) *
            g_ff(lam_a, t) *
            (loge10 / (2.0 * th * i_ev)) *
            10.0**(-th * i_ev))

def kappa_h_bf(lam_a, t, i_ev=i_h_ev):
    lam_a = np.asarray(lam_a, dtype=float)
    t = np.asarray(t, dtype=float)
    th = theta_5040(t)
    a0 = alpha0_hydrogen_cgs()
    n0 = n0_from_lambda(lam_a)
    s = np.zeros_like(lam_a, dtype=float)
    for dn in (0, 1, 2):
        n = n0 + dn
        active = lam_a <= (n.astype(float)**2) * lam_lyman_a
        boltz = 10.0**(-th * chi_exc_h_ev(n.astype(float)))
        term = (g_bf(lam_a, n.astype(float)) / (n.astype(float)**3)) * boltz
        s += np.where(active, term, 0.0)
    x3 = chi_exc_h_ev((n0 + 3).astype(float))
    rem = (loge10 / (2.0 * th * i_ev)) * (10.0**(-th * x3) - 10.0**(-th * i_ev))
    return a0 * (lam_a**3) * (s + rem)

def kappa_h_neutral(lam_a, t):
    return kappa_h_bf(lam_a, t) + kappa_h_ff(lam_a, t)

hminus_bf_coeffs = np.array([+1.99654, -1.18267e-5, +2.64243e-6, -4.40524e-10, +3.23992e-14, -1.39568e-18, +2.78701e-23])

hminus_thresh_a = 16421.0
hminus_taper_start = 15000.0

def alpha_bf_hminus(lam_a):
    lam_a = np.asarray(lam_a, dtype=float)
    p = np.zeros_like(lam_a)
    for k, ak in enumerate(hminus_bf_coeffs):
        p += ak * lam_a**k
    p *= 1e-18
    weight = np.ones_like(lam_a)
    mask = (lam_a >= hminus_taper_start) & (lam_a <= hminus_thresh_a)
    x = (lam_a[mask] - hminus_taper_start) / (hminus_thresh_a - hminus_taper_start)
    weight[mask] = np.sqrt(1 - x)
    weight[lam_a > hminus_thresh_a] = 0.0
    return np.clip(p * weight, 0.0, None)

def kappa_hminus_bf(lam_a, t, pe):
    lam_a = np.asarray(lam_a, dtype=float)
    t = np.asarray(t, dtype=float)
    pe = np.asarray(pe, dtype=float)
    th = theta_5040(t)
    a_bf = alpha_bf_hminus(lam_a)
    return 4.158e-10 * a_bf * pe * (th**2.5) * 10.0**(0.754 * th)

def f0_f1_f2_hminus_ff(lam_a):
    lam_a = np.asarray(lam_a, dtype=float)
    x = np.log10(lam_a)
    f0 = (-2.2763 - 1.6850*x + 0.76661*x**2 - 0.053346*x**3)
    f1 = (+15.2827 - 9.2846*x + 1.99381*x**2 - 0.142631*x**3)
    f2 = (-197.789 + 190.266*x - 67.9775*x**2 + 10.6913*x**3 - 0.625151*x**4)
    return f0, f1, f2

def kappa_hminus_ff(lam_a, t, pe):
    lam_a = np.asarray(lam_a, dtype=float)
    t = np.asarray(t, dtype=float)
    pe = np.asarray(pe, dtype=float)
    th = theta_5040(t)
    logth = np.log10(th)
    f0, f1, f2 = f0_f1_f2_hminus_ff(lam_a)
    expo = f0 + f1*logth + f2*(logth**2)
    return 1e-26 * pe * 10.0**(expo)