import numpy as np
import matplotlib.pyplot as plt
from astro530.opacity import kappa_hminus_bf, kappa_hminus_ff, kappa_h_neutral

T=11572
logPe=2.76
Pe = 10.0**logPe
lam = np.linspace(3000, 20000, 1000)  # Å

H = kappa_h_neutral(lam, T)/Pe
Hm_bf = kappa_hminus_bf(lam, T, Pe)/Pe
Hm_ff = kappa_hminus_ff(lam, T, Pe)/Pe
total = H + Hm_bf + Hm_ff

scale = 1e-26

plt.figure()
plt.plot(lam, total/scale, label="Total")
plt.plot(lam, Hm_bf/scale, "--", label="H- bf")
plt.plot(lam, Hm_ff/scale, ":", label="H- ff")
plt.plot(lam, H/scale, "-.", label="H")
plt.xlabel(r"$\lambda\ (\AA)$")
plt.ylabel(r"$\kappa_\lambda/P_e$  (unit = $10^{-26}$ cm$^2$ H$^{-1}$ dyne$^{-1}$)")
plt.legend()
ax = plt.gca()
ax.ticklabel_format(axis='y', style='plain')
ax.yaxis.get_offset_text().set_visible(False)
plt.savefig("8.5d.pdf", bbox_inches="tight")