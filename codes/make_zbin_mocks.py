"""
make_zbin_mocks.py -- completeness mocks for a z_a window that has no sky-completeness map of its own.

The mock tables of the two map bins (z_true 7.5-9.5 and 9.5-14.5) hold each injection's recovered Z_A and
the flag L3 (full selection, z_a anywhere in 8.0-13.5). Layer 4 of a window [lo, hi) is
L3 & (lo <= Z_A < hi): this reproduces the maps' own L4 column exactly (checked below for both tables).
The two tables are merged, so every true redshift that can scatter into the window is covered.

    python make_zbin_mocks.py parameters.yaml z8p0_9p0 z9p0_13p5

Writes paths.completeness of each named z-bin (with its per-bin override in parameters.yaml), the file
run_pipeline.load_completeness reads. The sky map (where on the sky completeness is higher or lower) is
not rebuilt: the z-bin points paths.sky_map at the nearest existing map and sets sky_map_approx.
"""
import sys
from pathlib import Path

import numpy as np
import yaml
from astropy.table import Table, vstack

PROJECT_DIR = Path(__file__).resolve().parent.parent
MAPS = PROJECT_DIR / "inputs/depth_maps/CEERS_SkyCompleteness_zbins/CEERS_SkyCompleteness_zbins"
SOURCES = {"z7p5_9p5": (8.0, 9.5), "z9p5_14p5": (9.5, 13.5)}      # map bin -> its own layer-4 z_a window
KEEP = ["RA", "DEC", "X", "Y", "MAG_IN", "Z_TRUE", "MUV_TRUE", "Z_A", "L3"]


def load_sources():
    """Both mock tables (columns KEEP + SOURCE), after checking L4 = L3 & z_a in the map's window."""
    tables = []
    for name, (lo, hi) in SOURCES.items():
        t = Table.read(MAPS / name / f"ceers_skycompleteness_mocks_{name}.fits")
        za, L3, L4 = np.asarray(t["Z_A"], float), np.asarray(t["L3"], bool), np.asarray(t["L4"], bool)
        bad = int(np.sum(L4 != (L3 & (za >= lo) & (za < hi))))
        if bad:
            raise ValueError(f"{name}: L4 != L3 & Z_A in [{lo}, {hi}) for {bad} mocks, so L4 cannot be rebuilt this way")
        t = t[KEEP]
        t["SOURCE"] = name
        tables.append(t)
    return vstack(tables, metadata_conflicts="silent")


def main(config_path, names):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    mocks = load_sources()
    za, L3, zt = (np.asarray(mocks[c]) for c in ("Z_A", "L3", "Z_TRUE"))
    for name in names:
        zb = cfg["survey"]["z_bins"][name]
        lo, hi = map(float, zb["z_a"])
        z_min, z_max = map(float, zb["z_true"])
        z_drop = round(0.5 * (z_min + z_max), 2)
        template = (zb.get("paths") or {}).get("completeness", cfg["paths"]["completeness"])
        out = PROJECT_DIR / template.format(z_bin=name, z_drop=z_drop)
        if out.resolve().is_relative_to(MAPS.resolve()):
            raise ValueError(f"{name}: refusing to write {out} inside the original map folder; give the z-bin its own paths.completeness")

        L4 = L3 & (za >= lo) & (za < hi)
        t = mocks.copy(copy_data=False)
        t["L4"] = L4
        t.meta.clear()
        t.meta.update(ZRANGE=f"{zt.min():.2f}-{zt.max():.2f}", L4WINDOW=f"{lo:.2f}-{hi:.2f}",
                      L4DEF="L3 and Z_A in [lo, hi)", SOURCES=",".join(SOURCES))
        out.parent.mkdir(parents=True, exist_ok=True)
        t.write(out, overwrite=True)
        inside = np.mean((zt[L4] >= z_min) & (zt[L4] <= z_max))
        print(f"{name}: z_a [{lo}, {hi}): {L4.sum()} of {len(t)} mocks pass; their z_true 1/50/99% = "
              f"{np.round(np.percentile(zt[L4], [1, 50, 99]), 2)}; {inside:.1%} inside z_true [{z_min}, {z_max}] "
              f"(mock-weighted)\n  -> {out.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python make_zbin_mocks.py parameters.yaml z_bin [z_bin ...]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2:])
