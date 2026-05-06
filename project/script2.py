import numpy as np
import pandas as pd
import splat
import matplotlib.pyplot as plt

rows = []

targets = [
    {"label": "Late-L", "query": {"opt_spt": ["L7", "L8"]}},
    {"label": "L/T transition", "query": {"opt_spt": ["T0", "T1"]}},
    {"label": "Early-T", "query": {"opt_spt": ["T2", "T3"]}},
    {"label": "Mid-T", "query": {"opt_spt": ["T5", "T6"]}},
]

spectra = []

for t in targets:
    try:
        splist = splat.getSpectrum(**t["query"])
        if len(splist) == 0:
            print(f"No spectra found for {t['label']}")
            continue
        sp = splist[0]   # pick first result for now
        spectra.append((t["label"], sp))
        print(f"Loaded {t['label']}: {sp.name}")
    except Exception as e:
        print(f"Failed for {t['label']}: {e}")

for label, sp in spectra:
    row = {"label": label, "name": sp.name}

    try:
        indices = splat.measureIndexSet(sp, set='burgasser')

        # save metadata separately if you want
        row["index_reference"] = indices.get("reference", "")
        row["index_bibcode"] = indices.get("bibcode", "")

        for key, entry in indices.items():
            # skip metadata keys
            if key in ["reference", "bibcode"]:
                continue

            print(f"{sp.name} | {key} -> {entry} | type={type(entry)}")

            # case 1: entry is tuple/list/array
            if isinstance(entry, (tuple, list, np.ndarray)):
                row[key] = entry[0] if len(entry) > 0 else np.nan
                row[key + "_err"] = entry[1] if len(entry) > 1 else np.nan

            # case 2: entry is dict
            elif isinstance(entry, dict):
                # try common field names
                row[key] = entry.get("value", entry.get("val", np.nan))
                row[key + "_err"] = entry.get("error", entry.get("err", np.nan))

            # case 3: entry is scalar
            else:
                row[key] = entry
                row[key + "_err"] = np.nan

    except Exception as e:
        print(f"Index measurement failed for {sp.name}: {e}")

    rows.append(row)

df = pd.DataFrame(rows)
print(df.T)
df.to_csv("lt_indices.csv", index=False)

plot_df = df.copy()
plot_df["seq"] = [0, 1, 2, 3]

index_cols = ["CH4-H", "CH4-K", "H2O-H"]

for col in index_cols:
    plt.figure(figsize=(6,4))
    plt.errorbar(
        plot_df["seq"],
        plot_df[col],
        yerr=plot_df.get(col + "_err", None),
        marker='o',
        capsize=4
    )
    plt.xticks([0,1,2,3], ["Late-L", "L/T", "Early-T", "Mid-T"])
    plt.xlabel("Spectral sequence")
    plt.ylabel(col)
    plt.title(f"{col} across the L–T transition")
    plt.tight_layout()
    plt.savefig(f"{col}_trend.png", dpi=200)
    plt.show()