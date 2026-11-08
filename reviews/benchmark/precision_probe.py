"""One-turn numerical controls for the selected ring, without timing claims."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

import common

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'workers'))
import w_slipstream  # noqa: E402
from slipstream.tracking.native.ring import RingMap  # noqa: E402


def main():
    torch.set_num_threads(1)
    case = common.load(HERE / 'cases/ring_tunes.json')
    lattice = common.load(HERE / 'cases' / case['lattice'])
    cfg = dict(device='cpu', dtype='float64', order=6, n_kicks=8, n_kicks_bend=16)
    base, _ = w_slipstream.build(lattice, case, cfg)
    initial = torch.tensor(common.launches(case), dtype=torch.float64)
    ref = np.load(common.RESULTS / '2026-10-27_momentum/ring_tunes/elegant/cpu-f64-o6s16b32/p0/tbt.npz')['tbt'][:, 1]
    rows = []
    for device, dtype, ds in [('cpu', torch.float64, False), ('cpu', torch.float32, False),
                               ('cpu', torch.float32, 'hybrid'), ('cpu', torch.float32, 'full'),
                               ('cuda', torch.float32, False), ('cuda', torch.float64, False)]:
        ring = RingMap(base.specs, case['energy_eV'], case['mass_eV'], n_kicks=8, n_kicks_bend=16,
                       integration_order=6, device=device, dtype=dtype, double_single=ds)
        result = ring.track(initial, 1, aperture=(1, 1)).state.cpu().numpy()[:, :4]
        assert np.isfinite(result).all()
        rows.append(dict(device=device, dtype=str(dtype), double_single=ds,
                         map_components=np.abs(result-ref).max(axis=0).tolist(), output=result.tolist()))
    out = dict(case_sha256=common.sha256(HERE / 'cases/ring_tunes.json'),
               lattice_sha256=common.sha256(HERE / 'cases' / case['lattice']),
               source_sha256=common.sha256(__file__), reference='2026-10-27_momentum/elegant/cpu-f64-o6s16b32/p0', rows=rows)
    dest = common.EVIDENCE / '2026-10-26_R1/precision_probe.json'
    dest.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps([{k: v for k, v in r.items() if k != 'output'} for r in rows], indent=2))


if __name__ == '__main__':
    main()
