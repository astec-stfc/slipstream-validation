"""Symplecticity and phase-space volume of the Slipstream ring map, against the Hamiltonian condition itself (no other code).

Slipstream's state is (x, px, y, py, tau, delta): tau [m] is the late-positive arrival coordinate and delta = (p - p0) / p0.  The
canonical partner of tau is minus E / (p0 c), not delta, so a one-turn Jacobian M becomes M_c = T M T^-1 with
T = diag(1, 1, 1, 1, 1, beta(delta)) and must satisfy M_c^T J M_c = J, where J is [[0, s], [-s, 0]] per pair with s = +1, +1, -1.
A determinant of 1 is necessary and not sufficient: a kick that is not the gradient of a potential keeps the volume.
"""
import math
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("slipstream")

from slipstream.tracking.native.lattice_import import ElementSpec as E  # noqa: E402
from slipstream.tracking.native.ring import RingMap  # noqa: E402

pytestmark = pytest.mark.slipstream
BENCH = pathlib.Path(__file__).resolve().parents[1] / "benchmark"
F64 = torch.float64
ENERGIES_EV = [3e9, 5e6]   # ring energy, and one where beta differs from 1 by 5e-3

EXACT = 1e-12   # target: float64 round-off, with a margin of about 100 on the worst measured element (4e-14, the benchmark ring)

# Launch states off axis and off momentum, where every term of a map is exercised.
STATES = torch.tensor([[2e-3, 3e-4, -1e-3, 2e-4, 0.0, 1e-2], [-4e-3, -1e-4, 2e-3, -3e-4, 1e-3, -2e-2]], dtype=F64)

SYMPLECTIC = {
    "drift": E("D", "drift", 1.0),
    "quad": E("Q", "quad", 0.3, k1=2.0),
    "sextupole": E("S", "sext", 0.2, k2=40.0),
    "octupole": E("O", "oct", 0.2, k3=3000.0),
    "skew_quad": E("SQ", "mult", 0.2, skew_multipoles=((1, 0.7),)),
    "thin_multipole": E("M", "mult", 0.0, integrated_multipoles=((2, 3.0),)),
    "misaligned_quad": E("QM", "quad", 0.3, k1=2.0, dx=1e-3, dy=-5e-4, tilt=0.05),
    "sector_bend": E("B", "bend", 1.0, angle=0.2),
    "rf_folded_to_drift": E("C", "rfca", 0.3, voltage_V=1e6, phase_rad=1.0, frequency_Hz=5e8),
    "corrector_off": E("K", "vkick", 0.2),
}

# Maps whose linear part about the reference orbit is symplectic, while terms of higher order are not.
LINEAR_ONLY = {
    "edge_bend": E("B", "bend", 1.0, angle=0.2, e1=0.1, e2=0.1),
    "combined_function_bend": E("B", "bend", 1.0, angle=0.2, k1=0.4, k2=3.0),
    "deflecting_cavity": E("T", "tdc", 0.2, voltage_V=1e6, phase_rad=0.3, frequency_Hz=3e9),
}

# Ratchets (about twice the value measured at STATES on 28 November 2026) for approximations that are intended and documented in Slipstream:
# the elegant-style edge map acts on slopes (EDGE_EFFECTS=1), and the deflecting cavity is symplectic with beta0 in place of beta(delta).
RATCHET = [   # (id, element, energy [eV], limit)
    ("edge_bend", LINEAR_ONLY["edge_bend"], 3e9, 3.3e-6),
    ("fringe_bend", E("B", "bend", 1.0, angle=0.2, e1=0.1, e2=0.1, hgap=0.02, fint=0.5), 3e9, 6.4e-6),
    ("second_order_edge", E("B", "bend", 1.0, angle=0.2, e1=0.1, e2=0.1, edge_order=2), 3e9, 3.3e-6),
    ("deflecting_cavity", LINEAR_ONLY["deflecting_cavity"], 3e9, 2.4e-11),
    ("deflecting_cavity", LINEAR_ONLY["deflecting_cavity"], 5e6, 6.6e-3),
]


def ring_of(specs, energy_eV=3e9, **kw):
    return RingMap(list(specs), energy_eV, device="cpu", dtype=F64, **{"n_kicks": 4, "n_kicks_bend": 8, **kw})


def canonical_jacobians(ring, states):
    """M_c, (N, 6, 6), of one turn at each of the N states.  Particles do not interact, so one backward pass per row serves them all."""
    x = states.clone().requires_grad_(True)
    y = ring.turn(x)
    M = torch.stack([torch.autograd.grad(y[:, i].sum(), x, retain_graph=True)[0] for i in range(6)], dim=1)
    p0 = math.sqrt(ring.energy_eV**2 - ring.mc2_eV**2)

    def beta(delta):
        p = p0 * (1 + delta)
        return p / torch.sqrt(p * p + ring.mc2_eV**2)

    out_scale, in_scale = torch.ones_like(states), torch.ones_like(states)
    out_scale[:, 5], in_scale[:, 5] = beta(y[:, 5].detach()), 1 / beta(states[:, 5])
    return out_scale[:, :, None] * M * in_scale[:, None, :]


def symplectic_error(M):
    """Largest element of |M^T J M - J| over a batch of canonical Jacobians."""
    J = torch.zeros(6, 6, dtype=F64)
    for k, sign in ((0, 1.0), (2, 1.0), (4, -1.0)):
        J[k, k + 1], J[k + 1, k] = sign, -sign
    return (M.transpose(-1, -2) @ J @ M - J).abs().max().item()


@pytest.mark.parametrize("energy_eV", ENERGIES_EV)
@pytest.mark.parametrize("name,exact_dipole", [*((name, True) for name in SYMPLECTIC), ("sector_bend", False)])
def test_element_is_symplectic(name, exact_dipole, energy_eV):
    M = canonical_jacobians(ring_of([SYMPLECTIC[name]], energy_eV, exact_dipole=exact_dipole), STATES)
    assert symplectic_error(M) < EXACT
    assert (torch.linalg.det(M) - 1).abs().max() < EXACT


@pytest.mark.parametrize("energy_eV", ENERGIES_EV)
@pytest.mark.parametrize("name", LINEAR_ONLY)
def test_linear_map_about_the_reference_orbit_is_symplectic(name, energy_eV):
    M = canonical_jacobians(ring_of([LINEAR_ONLY[name]], energy_eV), torch.zeros(1, 6, dtype=F64))
    assert symplectic_error(M) < EXACT


@pytest.mark.parametrize("spec,energy_eV,limit", [r[1:] for r in RATCHET], ids=[f"{r[0]}-{r[2]:g}" for r in RATCHET])
def test_documented_approximation_has_not_grown(spec, energy_eV, limit):
    assert symplectic_error(canonical_jacobians(ring_of([spec], energy_eV), STATES)) < limit


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "Slipstream defect, issue to be filed: the bend kick is dpx = -ds (1 + h x) Fy, dpy = ds (1 + h x) Fx, so d(dpx)/dy - d(dpy)/dx = "
    "-ds h Fx. It is not the gradient of a potential when h and K1 or K2 are both non-zero. The error grows with y and does not fall with "
    "more slices; the determinant stays 1."))
@pytest.mark.parametrize("strength", [dict(k1=0.4), dict(k2=3.0)], ids=["k1", "k2"])
def test_combined_function_bend_is_symplectic(strength):
    assert symplectic_error(canonical_jacobians(ring_of([E("B", "bend", 1.0, angle=0.2, **strength)]), STATES)) < EXACT


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "Slipstream, issue to be filed: a corrector with a non-zero kick follows elegant's matrix method on slopes, and its path length is taken "
    "from the entrance slopes only, so the time of flight lacks the term that pairs with the momentum dependence of the kick."))
@pytest.mark.parametrize("length", [0.0, 0.2], ids=["thin", "thick"])
def test_corrector_with_a_kick_is_symplectic(length):
    ring = ring_of([E("K", "hkick", length, kick_x=1e-3)])
    assert symplectic_error(canonical_jacobians(ring, torch.zeros(1, 6, dtype=F64))) < EXACT


def benchmark_ring():
    """The frozen ring of the cross-code benchmark, built by the benchmark's own Slipstream adapter at the paper's setting."""
    for path in (BENCH, BENCH / "workers"):
        sys.path.insert(0, str(path))
    import common
    import w_slipstream

    case = common.load(BENCH / "cases" / "ring_tunes.json")
    cfg = next(c for c in w_slipstream.CONFIGS if c["name"] == "cpu-f64-o6q8s1b8")
    return w_slipstream.build(common.load(BENCH / "cases" / case["lattice"]), case, cfg)[0]


def test_benchmark_ring_one_turn_map_is_symplectic():
    states = STATES * torch.tensor([1, 1, 1, 1, 1, 0.3], dtype=F64)   # within the ring's momentum acceptance
    M = canonical_jacobians(benchmark_ring(), states)
    assert symplectic_error(M) < EXACT
    assert (torch.linalg.det(M) - 1).abs().max() < EXACT


# ------------------------------------------------------------------------------------------------ many turns
QUADS = [E("QF", "quad", 0.2, k1=1.5), E("D1", "drift", 1.0), E("QD", "quad", 0.2, k1=-1.35), E("D2", "drift", 1.0)]
NONLINEAR = [*QUADS[:3], E("S", "sext", 0.1, k2=10.0), E("B", "bend", 0.5, angle=0.05), E("O", "oct", 0.1, k3=200.0), QUADS[3]]
TURNS = [2_000, pytest.param(100_000, marks=pytest.mark.slow)]


def orbit(ring, start, turns):
    """(turns + 1, N, 6): every state, tracked turn by turn."""
    record = torch.empty(turns + 1, *start.shape, dtype=F64)
    record[0] = u = start
    with torch.no_grad():
        for i in range(turns):
            record[i + 1] = u = ring.turn(u)
    return record


@pytest.mark.parametrize("turns", TURNS)
def test_volume_is_conserved_along_a_nonlinear_orbit(turns):
    """Liouville over many turns: the volume factor after T turns is the product of the one-turn determinants along the orbit."""
    ring = ring_of(NONLINEAR, n_kicks=1, n_kicks_bend=1, integration_order=4)
    start = torch.tensor([[1e-3, 0.0, 5e-4, 0.0, 0.0, 1e-3]], dtype=F64)
    states = orbit(ring, start, turns)[:-1, 0]
    assert torch.isfinite(states).all() and states[:, [0, 2]].abs().max() < 0.05   # a bounded orbit, or the check says nothing
    log_det, worst = 0.0, 0.0
    for chunk in states.split(20_000):
        M = canonical_jacobians(ring, chunk)
        log_det += torch.linalg.slogdet(M).logabsdet.sum().item()
        worst = max(worst, symplectic_error(M))
    assert worst < EXACT
    assert abs(log_det) < 1e-14 * turns   # target: no systematic volume change above 1e-14 per turn


@pytest.mark.parametrize("turns", TURNS)
def test_courant_snyder_invariant_does_not_drift(turns):
    """The tracked orbit itself (round-off included), where the invariant is known: a linear lattice at small amplitude."""
    ring = ring_of(QUADS, n_kicks=1, integration_order=4)
    M = torch.autograd.functional.jacobian(ring.turn, torch.zeros(6, dtype=F64))
    start = torch.tensor([[1e-6, 0, 1e-6, 0, 0, 0], [0, 3e-7, 0, -3e-7, 0, 0]], dtype=F64)
    record = orbit(ring, start, turns)
    t = torch.arange(turns + 1, dtype=F64) - turns / 2

    def drift(q):   # fitted relative change of the invariant over the whole run
        return ((t[:, None] * (q - q.mean(0))).sum(0) / (t**2).sum() * turns / q.mean(0)).abs().max().item()

    for i in (0, 2):
        B = M[i:i + 2, i:i + 2]
        half = (B[0, 0] - B[1, 1]) / 2
        S = torch.stack([torch.stack([-B[1, 0], half]), torch.stack([half, B[0, 1]])])   # sin(mu) [[gamma, alpha], [alpha, beta]]
        q = torch.einsum("tni,ij,tnj->tn", record[..., i:i + 2], S, record[..., i:i + 2])
        assert drift(q) < 1e-10   # target; measured 1.4e-13 over 2e4 turns
        assert drift(q * (1 - 1e-13 * (t[:, None] + turns / 2))) > 1e-10   # the gate can fail: damping of 1e-13 per turn is seen
