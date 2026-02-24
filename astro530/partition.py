from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np

THETA_GRID = np.arange(0.2, 2.01, 0.2)

@dataclass(frozen=True)
class PartitionRow:
    logU_theta: np.ndarray
    logg0: float

def _to_float_or_nan(tok):
    tok = tok.strip()
    if tok == "-" or tok.lower() == "nan":
        return float("nan")
    return float(tok)

def load_ioniz(path):
    out: Dict[str, dict] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            parts = line.split()
            Z = int(parts[0])
            sym = parts[1]
            weight = float(parts[2])
            chis = [float(x) for x in parts[3:]]
            out[sym] = {"Z": Z, "weight": weight, "chis_eV": chis}
    return out


def load_partition_table(path):
    table: Dict[str, PartitionRow] = {}
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            parts = line.split()
            species = parts[0]
            nums = parts[1:]
            vals = np.array([_to_float_or_nan(x) for x in nums], dtype=float)
            logU = vals[:10]
            logg0 = float(vals[10])
            table[species] = PartitionRow(logU_theta=logU, logg0=logg0)
    return table


def partition_function(species, T, part_table):
    theta = 5040.0 / T
    sp = species.strip()
    if sp in {"H-"}:
        return 1.0, 1.0
    if sp in {"H+"}:
        return 1.0, 1.0
    row = part_table[sp]
    g0 = 10.0 ** row.logg0
    th = float(theta)
    y = row.logU_theta
    m = np.isfinite(y)
    if not np.any(m):
        return float(g0), float(g0)
    x = THETA_GRID[m]
    y = y[m]
    if th <= x.min():
        logU = float(y[x.argmin()])
    elif th >= x.max():
        logU = float(y[x.argmax()])
    else:
        logU = float(np.interp(th, x, y))
    U = 10.0 ** logU
    return float(U), float(g0)

def saha_phi(T, I_eV, u_upper, u_lower):
    return 0.6665 * (u_upper / u_lower) * (T ** 2.5) * (10.0 ** (-5040.0 * I_eV / T))