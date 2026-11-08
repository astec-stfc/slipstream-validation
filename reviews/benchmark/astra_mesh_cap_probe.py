"""Find the effective ASTRA 3-D mesh size: the 96^3 and 128^3 sc_drift runs gave identical output (SLIPSTREAM_BENCHMARK_IMPROVEMENTS.md, P1).

Runs ASTRA on the first 0.3 m of the sc_drift bunch (ten times cheaper than the full 3 m) for several Nxf = Nyf = Nzf and reports how the rms
size, emittance and energy spread at the last station change between meshes, plus ASTRA's own echo of the namelist in its log (the echo shows what
was requested, not what was used).  Not a timing run.

    python astra_mesh_cap_probe.py
"""
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'workers'))
import common  # noqa: E402
from astra_deck import write_deck  # noqa: E402

EXE = pathlib.Path(os.environ.get('BENCH_ASTRA_EXE', 'Astra'))
MESHES = (48, 64, 72, 80, 88, 96, 128)
ZSTOP = 0.3


def run(n3d):
    case = common.load(HERE / 'cases/sc_drift.json')
    case = dict(case, stations_m=[s for s in case['stations_m'] if s <= ZSTOP] or [0.0, ZSTOP])
    d = pathlib.Path(tempfile.mkdtemp(prefix='bench_astra_probe_'))
    deck = write_deck(d, common.launches(case), case, 0, 0, 1.4e-3, True, n3d)
    subprocess.run([str(EXE), deck.name], cwd=d, capture_output=True, text=True, timeout=1800, input='\n')
    x = np.loadtxt(d / 'sc_drift.Xemit.001')[-1]
    z = np.loadtxt(d / 'sc_drift.Zemit.001')[-1]
    log = (d / 'sc_drift.Log.001').read_bytes().decode('latin-1')
    echoed = re.search(r'NXF\s*=\s*(\d+)', log)
    return dict(n3d=n3d, echoed_nxf=int(echoed.group(1)) if echoed else None, sigma_x_mm=float(x[3]), emit_x=float(x[5]), sigma_E_keV=float(z[4]), ekin_MeV=float(z[2]))


def main():
    rows = [run(n) for n in MESHES]
    for a, b in zip(rows, rows[1:]):
        b['change_vs_previous'] = {k: abs(b[k] - a[k]) / abs(a[k]) for k in ('sigma_x_mm', 'emit_x', 'sigma_E_keV')}
    for r in rows:
        print(r)
    out = dict(design=dict(date='2026-10-27', z_stop_m=ZSTOP, note='Short-drift probe of the effective ASTRA 3-D mesh; the full-drift 96^3 and 128^3 outputs were byte-identical'),
               analysis_sha256=common.sha256(__file__), rows=rows)
    (common.EVIDENCE / '2026-10-26_R1/astra_mesh_cap_probe.json').write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
