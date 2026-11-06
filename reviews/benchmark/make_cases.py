"""Freeze the code-neutral lattice files in cases/ (run once, commit the output; every adapter reads the frozen JSON).

    python make_cases.py            # needs slipstream and the slipstream_examples checkout (SLIPSTREAM_EXAMPLES overrides)

The ring is the illustrative 48 m, twofold-symmetric DBA lattice of slipstream_examples/nonlinear_optimisation.py,
not an operational MLS lattice.  The zero-strength octupole is stored as a drift (K3 = 0 in the example).
"""
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
EX = pathlib.Path(os.environ["SLIPSTREAM_EXAMPLES"])
sys.path.insert(0, str(EX))
sys.path.insert(0, str(HERE))
from common import lattice_inventory  # noqa: E402
from nonlinear_optimisation import mls_specs  # noqa: E402

els = []
for s in mls_specs():
    kind = "drift" if s.kind == "oct" else s.kind
    assert kind in ("drift", "quad", "sext", "bend"), kind
    e = dict(name=s.name, kind=kind, length=s.length)
    key = dict(quad="k1", sext="k2", bend="angle").get(kind)
    if key:
        e[key] = getattr(s, key)  # explicit even when zero (the S3 sextupoles are off)
    els.append(e)
lat = dict(id="mls_ring", note="illustrative 48 m MLS-inspired DBA ring, sector bends (e1 = e2 = 0), thick elements",
           source="slipstream_examples/nonlinear_optimisation.py:mls_specs()", elements=els)
lat["inventory"] = lattice_inventory(lat)
(HERE / "cases").mkdir(exist_ok=True)
(HERE / "cases" / "mls_ring.json").write_text(json.dumps(lat, indent=1))
print(lat["inventory"])

# ---------------------------------------------------------------------------------------- sc_drift: frozen macroparticles
# 5 MeV electron bunch, 20 pC, Gaussian in 3D, in the laboratory frame at one instant (the form ASTRA and GPT read; ImpactX takes it inside
# ``at_fixed_t``).  x, y, z in m (z > 0 is the head, relative to the reference particle), bg* = beta*gamma components of the momentum.
import numpy as np  # noqa: E402

MC2 = 0.51099895e6
rng = np.random.default_rng(20261022)
N, EK = 20000, 5.0e6
gamma0 = 1 + EK / MC2
bg0 = np.sqrt(gamma0**2 - 1)
sx, sz, epsn, sdelta = 0.5e-3, 1.0e-3, 0.3e-6, 1e-3
x, y, z = rng.normal(0, sx, N), rng.normal(0, sx, N), rng.normal(0, sz, N)
bgx, bgy = rng.normal(0, epsn / sx, N), rng.normal(0, epsn / sx, N)  # sigma_{beta gamma x} = eps_n / sigma_x (uncorrelated waist)
bgz = bg0 * (1 + rng.normal(0, sdelta, N))
for a in (x, y, z, bgx, bgy, bgz):  # remove the sampling offset so that all codes start from an exactly centred, zero-mean bunch
    a -= a.mean() if a is not bgz else a.mean() - bg0
np.savez_compressed(HERE / "cases" / "sc_drift.npz", x=x, y=y, z=z, bgx=bgx, bgy=bgy, bgz=bgz)
(HERE / "cases" / "sc_drift_lat.json").write_text(json.dumps(dict(
    id="sc_drift_lat", note="field-free drift", elements=[dict(name="D", kind="drift", length=3.0)]), indent=1))
print("sc_drift:", dict(N=N, bg0=bg0, sigma_x=float(x.std()), sigma_z=float(z.std()), eps_n=float(np.sqrt(x.var() * bgx.var() - np.mean(x * bgx) ** 2))))
