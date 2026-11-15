"""Regenerate the large particle files of the sc_drift particle-count ladder (200 000 and 1 000 000 macroparticles; 2 000 is tracked in cases/).

They are not kept in git (9 MB and 45 MB).  The recipe is that of ``injector_scaling_control.make_cases`` (fixed seeds), but only the .npz files are written, so the
tracked case json files (their hashes are in every archived result) stay untouched.  Each regenerated file is compared with the SHA-256 recorded by the archived
runs of ``2026-10-28_inj_scaling`` (10 November 2026: all three were byte-identical).  Add the two files to ``.git/info/exclude`` to keep them out of ``git status``.

    python regen_ladder_particles.py            # writes cases/sc_drift_n200000.npz and cases/sc_drift_n1000000.npz, checks the hashes
"""
import hashlib
import json

import numpy as np

import common

CASES = common.ROOT / "cases"
ARCHIVE = common.RESULTS / "2026-10-28_inj_scaling"
MC2, EK = 0.51099895e6, 5.0e6
SX, SZ, EPSN, SDELTA = 0.5e-3, 1.0e-3, 0.3e-6, 1e-3


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    bg0 = np.sqrt((1 + EK / MC2) ** 2 - 1)
    for n in (200000, 1000000):
        rng = np.random.default_rng(20261028 + n)
        x, y, z = rng.normal(0, SX, n), rng.normal(0, SX, n), rng.normal(0, SZ, n)
        bgx, bgy = rng.normal(0, EPSN / SX, n), rng.normal(0, EPSN / SX, n)
        bgz = bg0 * (1 + rng.normal(0, SDELTA, n))
        for a in (x, y, z, bgx, bgy, bgz):
            a -= a.mean() if a is not bgz else a.mean() - bg0
        out = CASES / f"sc_drift_n{n}.npz"
        np.savez_compressed(out, x=x, y=y, z=z, bgx=bgx, bgy=bgy, bgz=bgz)
        archived = None
        for r in sorted(ARCHIVE.glob(f"sc_drift_n{n}/slipstream/*/p0/result.json")):
            archived = json.loads(r.read_text()).get("particles_sha256")
            break
        h = sha(out)
        print(f"{out.name}: {h[:16]}  archived {str(archived)[:16]}  ->", "identical" if archived is None or h == archived else "DIFFERENT (numpy or zlib version?)")


if __name__ == "__main__":
    main()
