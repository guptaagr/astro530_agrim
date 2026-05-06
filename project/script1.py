import numpy as np
import matplotlib.pyplot as plt
import splat

# --------------------------------------------------
# 1) Choose a few representative objects
# Replace names/shortnames later if needed
# --------------------------------------------------
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

# --------------------------------------------------
# 2) Normalize each spectrum near 1.27 microns
# --------------------------------------------------
def normalize_spectrum(wave, flux, lo=1.26, hi=1.28):
    mask = (wave >= lo) & (wave <= hi) & np.isfinite(flux)
    if np.sum(mask) == 0:
        return flux
    scale = np.nanmedian(flux[mask])
    return flux / scale if scale != 0 else flux

# --------------------------------------------------
# 3) Plot
# --------------------------------------------------
plt.figure(figsize=(10, 6))

for label, sp in spectra:
    wave = np.array(sp.wave)
    flux = np.array(sp.flux)
    good = np.isfinite(wave) & np.isfinite(flux)
    wave = wave[good]
    flux = flux[good]
    flux_norm = normalize_spectrum(wave, flux)

    plt.plot(wave, flux_norm, lw=1.2, label=f"{label}: {sp.name}")

# Mark important regions
for x, txt in [(1.15, "H$_2$O"), (1.40, "H$_2$O"), (1.60, "CH$_4$"),
               (1.90, "H$_2$O"), (2.20, "CH$_4$")]:
    plt.axvline(x, ls="--", alpha=0.3)
    plt.text(x, 0.15, txt, rotation=90, va="bottom", ha="center", fontsize=9)

plt.xlim(0.9, 2.45)
plt.ylim(0, 1.8)
plt.xlabel("Wavelength [micron]")
plt.ylabel("Normalized flux")
plt.title("Representative L–T dwarf spectra")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("lt_sequence_overlay.png", dpi=200)
plt.show()