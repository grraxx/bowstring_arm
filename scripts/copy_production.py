"""
copy_production.py
One-off: copy the fields of ~/elastica/data/production (d 0.70-0.99, phi +-pi/2, step 0.01)
into data/production/fields.npz, so this project no longer reads ~/elastica/data.
theta(s) (2.4 GB) and the tetrahedra are left out: the segment table needs only the fields.
"""
import sys
from pathlib import Path
import numpy as np

src = Path.home() / "elastica/data/production/solution_space_results.npz"
dst = Path(__file__).resolve().parents[1] / "data/production/fields.npz"
keys = ("shape", "d_vals", "phi1_vals", "phi2_vals", "beam_x", "fx", "fy", "moment",
        "moment_right", "energy", "p", "q", "branch", "solved", "promoted", "waited")
with np.load(src) as z:
    out = {k: z[k] for k in keys}
shape = tuple(out["shape"])
for k in keys[5:]:
    out[k] = out[k].reshape(shape)
np.savez_compressed(dst, source=str(src), **out)
print(f"wrote {dst} ({dst.stat().st_size / 1e6:.0f} MB), grid {shape}")
