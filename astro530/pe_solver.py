from __future__ import annotations

import numpy as np
import pandas as pd
from astro530.partition import saha_phi, partition_function

def load_solar_abundances(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    df.columns = ["atomic", "element", "weight", "A", "logA", "logA12"]

    df["atomic"] = pd.to_numeric(df["atomic"], errors="coerce").astype("Int64")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df["A"] = pd.to_numeric(df["A"], errors="coerce")
    df["logA"] = pd.to_numeric(df["logA"], errors="coerce")
    df["logA12"] = pd.to_numeric(df["logA12"], errors="coerce")

    return df


def available_species_for_pe(ioniz: dict, part_table: dict, abund_df: pd.DataFrame,
                             partition_function, test_T: float = 6000.0) -> list[str]:
    out = []

    for _, row in abund_df.iterrows():
        sym = str(row["element"]).strip()
        A = row["A"]

        if pd.isna(A) or A <= 0:
            continue
        if sym not in ioniz:
            continue
        if len(ioniz[sym]["chis_eV"]) < 1:
            continue

        try:
            partition_function(sym, test_T, part_table)
            partition_function(sym + "+", test_T, part_table)
        except Exception:
            continue

        out.append(sym)

    return out


def phi_first_ionization(symbol: str, T: float, ioniz: dict, part_table: dict, partition_function) -> float:
    I1 = ioniz[symbol]["chis_eV"][0]
    u0, _ = partition_function(symbol, T, part_table)
    u1, _ = partition_function(symbol + "+", T, part_table)
    return saha_phi(T, I1, u_upper=u1, u_lower=u0)


def gray_eq_9p8_rhs(Pe: float, Pg: float, T: float, species: list[str],
                    abund_df: pd.DataFrame, ioniz: dict, part_table: dict,
                    partition_function) -> float:
    if Pe <= 0:
        raise ValueError("Pe must be positive.")

    a_lookup = abund_df.set_index("element")["A"].to_dict()

    num = 0.0
    den = 0.0

    for sym in species:
        Aj = float(a_lookup[sym])
        phij = phi_first_ionization(sym, T, ioniz, part_table, partition_function)
        r = phij / Pe

        frac_ion = r / (1.0 + r)   # n1 / (n0+n1)
        num += Aj * frac_ion
        den += Aj * (1.0 + frac_ion)

    return Pg * num / den


def initial_pe_guess(Pg: float, T: float) -> float:
    if T < 5000:
        f = 1e-4
    elif T < 6500:
        f = 3e-4
    elif T < 8000:
        f = 1e-3
    elif T < 10000:
        f = 3e-3
    else:
        f = 1e-2

    return min(max(f * Pg, 1e-12), 0.5 * Pg)


def solve_pe_gray_9p8(
    T: float,
    Pg: float,
    abund_df: pd.DataFrame,
    ioniz: dict,
    part_table: dict,
    partition_function,
    tol: float = 1e-8,
    max_iter: int = 200,
    damping: float = 0.5,
    Pe0: float | None = None,
    verbose: bool = False,
):
    """
    Solve Gray Eq. (9.8) by damped fixed-point iteration.

    Returns
    -------
    Pe : float
        Converged electron pressure.
    history : list[float]
        Iteration history of Pe values.
    """
    species = available_species_for_pe(ioniz, part_table, abund_df, partition_function)
    print("Species used:", species)
    if len(species) == 0:
        raise ValueError("No overlapping species found between abundances, ionization table, and partition table.")

    Pe = initial_pe_guess(Pg, T) if Pe0 is None else float(Pe0)
    history = [Pe]

    for it in range(max_iter):
        rhs = gray_eq_9p8_rhs(Pe, Pg, T, species, abund_df, ioniz, part_table, partition_function)

        Pe_new = (1.0 - damping) * Pe + damping * rhs
        history.append(Pe_new)

        rel = abs(Pe_new - Pe) / Pe
        if verbose:
            print(f"iter={it:3d}  Pe={Pe:.6e}  rhs={rhs:.6e}  Pe_new={Pe_new:.6e}  rel={rel:.3e}")

        if rel < tol:
            return Pe_new, history

        Pe = Pe_new

    raise RuntimeError("Pe iteration did not converge within max_iter.")


def summarize_species_used(abund_df: pd.DataFrame, ioniz: dict, part_table: dict) -> pd.DataFrame:
    species = available_species_for_pe(ioniz, part_table, abund_df, partition_function)
    sub = abund_df[abund_df["element"].isin(species)][["atomic", "element", "weight", "A", "logA", "logA12"]].copy()
    return sub.sort_values("atomic").reset_index(drop=True)