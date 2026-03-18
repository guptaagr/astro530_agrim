import numpy as np
import pandas as pd
from astro530.partition import load_ioniz, load_partition_table, partition_function
from astro530.pe_solver import load_solar_abundances, solve_pe_gray_9p8, summarize_species_used

ioniz = load_ioniz("../hw6/ioniz.txt")
part = load_partition_table("../hw6/RepairedPartitionFunctions.txt")
abund = load_solar_abundances("SolarAbundance.txt")

# (a)
print(summarize_species_used(abund, ioniz, part))

T = 6429
logPg = 5.1
Pg = 10.0**logPg

Pe, hist = solve_pe_gray_9p8(
    T=T,
    Pg=Pg,
    abund_df=abund,
    ioniz=ioniz,
    part_table=part,
    partition_function=partition_function,
    tol=1e-8,
    max_iter=200,
    damping=0.5,
    verbose=True,
)

print(f"\nConverged Pe = {Pe:.6e}")
print(f"log Pe = {np.log10(Pe):.4f}")

# (c) and (d)

def abundance_sums(abund_df):
    df = abund_df.copy()

    df["atomic"] = pd.to_numeric(df["atomic"], errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df["A"] = pd.to_numeric(df["A"], errors="coerce")

    light30 = df[(df["atomic"].notna()) & (df["atomic"] <= 30) & (df["A"].notna()) & (df["weight"].notna())]
    metals = df[(df["atomic"].notna()) & (df["atomic"] > 2) & (df["A"].notna()) & (df["weight"].notna())]

    sum_A_light30 = light30["A"].sum()
    sum_A_mu_light30 = (light30["A"] * light30["weight"]).sum()

    sum_A_metals = metals["A"].sum()
    sum_A_mu_metals = (metals["A"] * metals["weight"]).sum()

    return {
        "light30": {
            "sum_A": float(sum_A_light30),
            "sum_A_mu": float(sum_A_mu_light30),
        },
        "metals": {
            "sum_A": float(sum_A_metals),
            "sum_A_mu": float(sum_A_mu_metals),
        },
    }


# example usage
sums = abundance_sums(abund)
print("Lightest 30 elements:")
print("sum A_j =", sums["light30"]["sum_A"])
print("sum A_j mu_j =", sums["light30"]["sum_A_mu"], "amu")

print("\nMetals only (Z > 2):")
print("sum A_j =", sums["metals"]["sum_A"])
print("sum A_j mu_j =", sums["metals"]["sum_A_mu"], "amu")