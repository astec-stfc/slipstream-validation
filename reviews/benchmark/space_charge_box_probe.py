"""Independent static Gaussian check of Cartesian box size and field resolution."""
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.special import erf

import common
from slipstream.constants import EPS0_F_PER_M
from slipstream.tracking.native import space_charge
from slipstream.tracking.native.space_charge import SpaceCharge3D

HERE = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1)
    sigma, charge = 1e-3, 1e-12
    r = sigma * np.array([0.5, 1.0, 2.0, 3.0])
    expected = charge / (4 * math.pi * EPS0_F_PER_M * r**2) * (
        erf(r / (math.sqrt(2) * sigma)) - math.sqrt(2 / math.pi) * r / sigma * np.exp(-r**2 / (2*sigma**2)))
    rows = []
    # The first two have equal physical cell spacing; the third refines that spacing at fixed extent.
    for n, half_sigmas in ((17, 4), (25, 6), (33, 4)):
        half = half_sigmas * sigma
        h = 2 * half / (n - 1)
        coords = (torch.arange(n, dtype=torch.float64) - (n - 1) / 2) * h
        x, y, z = torch.meshgrid(coords, coords, coords, indexing='ij')
        density = charge / ((2*math.pi)**1.5 * sigma**3) * torch.exp(-(x*x + y*y + z*z) / (2*sigma**2))
        node_charge = density * h**3
        solver = SpaceCharge3D(n, n, n, h, h, h, -half, -half, -half, device='cpu', dtype=torch.float64)
        grids = solver.solve(node_charge)
        rr = torch.tensor(r, dtype=torch.float64)
        zero = torch.zeros_like(rr)
        field = solver.gather(grids[0], rr, zero, zero).numpy()
        cross = solver.gather(grids[1], rr, zero, zero).numpy()
        assert np.isfinite(field).all() and np.max(np.abs(cross)) < np.max(expected) * 1e-10
        rows.append(dict(n=n, half_box_sigma=half_sigmas, cell_sigma=h/sigma,
                         charge_fraction=float(node_charge.sum()/charge), field=field.tolist(),
                         relative_error=np.abs(field / expected - 1).tolist()))
    assert max(rows[0]['relative_error']) < 0.15 and max(rows[2]['relative_error']) < 0.05
    assert np.max(np.abs((np.array(rows[0]['field']) - rows[1]['field']) / expected)) < 1e-3
    out = dict(method='Static spherical Gaussian grid density; analytic free-space radial electric field. '
                      'No particles, time stepping, Lorentz transform or fitted injector box.',
               sigma_m=sigma, charge_C=charge, radius_sigma=(r/sigma).tolist(), expected_V_per_m=expected.tolist(),
               source_sha256=common.sha256(__file__), solver_sha256=common.sha256(space_charge.__file__),
               torch_version=torch.__version__, device='cpu', dtype='float64', rows=rows)
    dest = common.EVIDENCE / '2026-10-26_R1/space_charge_box_probe.json'
    dest.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
