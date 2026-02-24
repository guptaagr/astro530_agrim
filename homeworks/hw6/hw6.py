from astro530.partition import load_ioniz, load_partition_table, partition_function, saha_phi
import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import k_B
import astropy.units as u

# Problem 5
ioniz = load_ioniz("ioniz.txt")
part_table = load_partition_table("RepairedPartitionFunctions.txt")

U_Fe, g0_Fe = partition_function("Fe", 6000.0, part_table)
U_Hm, g0_Hm = partition_function("H-", 6000.0, part_table)
U_HII, g0_HII = partition_function("H+", 6000.0, part_table)
print(U_Fe, g0_Fe)
print(U_Hm, g0_Hm)
print(U_HII, g0_HII)

#Problem 6, Fig 1
T = np.linspace(4000.0, 8000.0, 400) * u.K
chi_12 = 10.20 * u.eV
U_H, g1 = partition_function("H", 6000.0, part_table)
g2 = 8.0
ratio = (g2 / g1) * np.exp(-(chi_12 / (k_B * T)).decompose().value)
f1 = 1.0 / (1.0 + ratio)
f2 = ratio / (1.0 + ratio)

plt.figure()
plt.plot(T.value, np.log10(f1), linewidth=2)
plt.plot(T.value, np.log10(f2), linewidth=2)
plt.xlabel("Temperature (K)")
plt.ylabel(r"$\log_{10}(N_n/N)$")
plt.ylim(-14, 1)
plt.xlim(3000, 9000)
plt.title("Fig 1.9-like: H excitation (n=1,2)")
plt.savefig("hw6_excitation.svg")

#Problem 6, Fig 2
element = "Fe"
T = np.linspace(2000.0, 10000.0, 600)
Pe = 1.0
chis = ioniz[element]["chis_eV"]
I1 = chis[0]
I2 = chis[1]
f0 = np.zeros_like(T)
f1 = np.zeros_like(T)
f2 = np.zeros_like(T)

for i, Ti in enumerate(T):
    u0, _ = partition_function("Fe", Ti, part_table)
    u1, _ = partition_function("Fe+", Ti, part_table)
    u2, _ = partition_function("Fe++", Ti, part_table)
    Phi1 = saha_phi(Ti, I1, u_upper=u1, u_lower=u0)
    Phi2 = saha_phi(Ti, I2, u_upper=u2, u_lower=u1)
    a = Phi1 / Pe
    b = Phi2 / Pe
    ab = a * b
    denom = 1.0 + a + ab
    f0[i] = 1.0 / denom
    f1[i] = a / denom
    f2[i] = ab / denom
plt.figure()
plt.plot(T, f0, linestyle="--", linewidth=2, label="Neutral (Fe)")
plt.plot(T, f1, linestyle="-",  linewidth=2, label="1st ion (Fe+)")
plt.plot(T, f2, linestyle=":",  linewidth=2, label="2nd ion (Fe++)")
plt.xlabel("Temperature (K)")
plt.ylabel("Fraction in ionization stage")
plt.xlim(2000, 10000)
plt.ylim(-0.02, 1.05)
plt.legend()
plt.title("Fig 1.10-like: Iron ionization fractions (Pe = 1 dyne/cm²)")
plt.savefig("hw6_ionization.svg")