"""Steady-state 1-D CSR of a compression spike: 20 % of the charge in a core ten times shorter than the rest of the bunch.

Reference: the unshielded steady-state wake of Saldin, Schneidmiller and Yurkov by adaptive quadrature, as in Slipstream's own
test of a smooth Gaussian,  delta(tau) = 2 L / (4 pi eps0 (3 rho^2)^(1/3) E) * int_0^inf lambda'(tau + u) u^(-1/3) du.
The particles sit at the quantiles of the density, so only the error of the estimator remains.  A 1-D model is the reference here:
where the 1-D assumption itself fails (transverse size) is not tested.
"""
import functools
import math

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("slipstream")
scipy_quad = pytest.importorskip("scipy.integrate").quad
ndtri = pytest.importorskip("scipy.special").ndtri

from slipstream.constants import EPS0_F_PER_M  # noqa: E402
from slipstream.tracking.native.collective import csr_steady_state_delta_kick, csr_steady_state_delta_kick_fast  # noqa: E402

pytestmark = pytest.mark.slipstream
ENERGY_EV, CHARGE_C, LENGTH_M, RHO_M = 190e6, 100e-12, 0.2, 4.0
PARTS = ((0.8, 20e-6), (0.2, 2e-6))   # (fraction of the charge, rms length [m])
CORE_M = 3 * PARTS[0][1]


def bunch(n):
    return np.concatenate([s * ndtri((np.arange(round(f * n)) + 0.5) / round(f * n)) for f, s in PARTS])


@functools.lru_cache(maxsize=None)
def reference_on_grid():
    def slope(v):
        return sum(-v / s**2 * f * CHARGE_C / (math.sqrt(2 * math.pi) * s) * math.exp(-0.5 * (v / s) ** 2) for f, s in PARTS)

    scale = 2 * LENGTH_M / (4 * math.pi * EPS0_F_PER_M * (3 * RHO_M**2) ** (1 / 3) * ENERGY_EV)

    def at(tau):   # u = t**1.5 removes the singularity: u**(-1/3) du = 1.5 dt
        top = (max(12 * PARTS[0][1] - tau, 0.0) + PARTS[0][1]) ** (2 / 3)
        marks = sorted({u ** (2 / 3) for _, s in PARTS for u in (-tau - 2 * s, -tau, -tau + 2 * s) if u > 0})
        return 1.5 * scale * scipy_quad(lambda t: slope(tau + t**1.5), 0, top, points=[m for m in marks if m < top] or None,
                                        limit=400, epsabs=0, epsrel=1e-10)[0]

    grid = np.unique(np.concatenate([np.linspace(-6 * s, 6 * s, 1201) for _, s in PARTS]))
    return grid, np.array([at(t) for t in grid])


def errors(tau, kick):
    """(rms, largest) error over the core, relative to the largest exact kick."""
    grid, ref = reference_on_grid()
    diff = (kick - np.interp(tau, grid, ref))[np.abs(tau) < CORE_M] / np.abs(ref).max()
    return float(np.sqrt(np.mean(diff**2))), float(np.abs(diff).max())


def grid_kick(tau, bins):
    energy = torch.tensor(ENERGY_EV, dtype=torch.float64)
    return csr_steady_state_delta_kick_fast(torch.tensor(tau), CHARGE_C / len(tau), RHO_M, energy, LENGTH_M, n_bins=bins).numpy()


def test_grid_kernel_converges_on_the_spike():
    tau = bunch(200_000)
    found = [errors(tau, grid_kick(tau, bins)) for bins in (256, 512, 1024, 2048)]
    for coarse, fine in zip(found, found[1:]):
        assert fine[0] < 0.4 * coarse[0] and fine[1] < 0.4 * coarse[1]   # target: second order, or close to it
    assert found[-1][0] < 0.002 and found[-1][1] < 0.005                 # ratchet: measured 0.0009 and 0.0024


def test_grid_kernel_at_the_default_256_bins():
    rms, largest = errors(bunch(200_000), grid_kick(bunch(200_000), 256))
    assert rms < 0.09 and largest < 0.26   # ratchet: measured 0.044 and 0.128


@pytest.mark.parametrize("n", [3000, 10000])
def test_grid_kernel_at_equal_particle_number(n):
    """The fair comparison with the pairwise kernel below: the same particles, drawn at random so that shot noise counts."""
    rng = np.random.default_rng(0)
    tau = np.concatenate([s * rng.standard_normal(round(f * n)) for f, s in PARTS])
    assert errors(tau, grid_kick(tau, 256))[0] < 0.11   # ratchet: mean rms over five seeds 0.055 at 3000 particles, 0.038 at 10 000


@pytest.mark.parametrize("n", [3000, 10000])
def test_pairwise_kernel_on_the_spike(n):
    """The kernel of ``fast_collective_effects=False`` (the LinacTracker default) smooths with Silverman's bandwidth of the whole bunch,
    far wider than the spike.  The limits record how far that is from the reference (measured rms 0.24 and 0.22, largest 0.63 and 0.57, a
    quarter above for the ratchet).  They are not a target, and more particles help little."""
    tau = bunch(n)
    energy = torch.tensor(ENERGY_EV, dtype=torch.float64)
    rms, largest = errors(tau, csr_steady_state_delta_kick(torch.tensor(tau), CHARGE_C / n, RHO_M, energy, LENGTH_M).numpy())
    assert rms < 0.30 and largest < 0.80


@pytest.mark.parametrize("kernel", [csr_steady_state_delta_kick, csr_steady_state_delta_kick_fast])
@pytest.mark.parametrize("tau", [[0.0] * 1000, [0.0, 1e-6]], ids=["zero_length", "two_particles"])
def test_degenerate_bunch_gives_no_nan(kernel, tau):
    """The 1-D kick of a bunch of no length diverges, so only finiteness is asked; see the README for the size."""
    energy = torch.tensor(ENERGY_EV, dtype=torch.float64)
    kick = kernel(torch.tensor(tau, dtype=torch.float64), CHARGE_C / len(tau), RHO_M, energy, LENGTH_M)
    assert torch.isfinite(kick).all()
