"""Particles that graze apertures and collimators.

The injector tests the straight path of each step against every aperture.  The oracle samples the same path densely: a path that has a
sample inside the slab and outside the acceptance must be lost, and a loss the samples do not see must be confirmed by finer ones.
"""
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("slipstream")

from slipstream.tracking.native.injector import _aperture_hit, _apply_apertures  # noqa: E402
from slipstream.tracking.native.lattice_import import ElementSpec as E  # noqa: E402
from slipstream.tracking.native.magnets import placement, to_local  # noqa: E402
from slipstream.tracking.native.ring import RingMap  # noqa: E402

pytestmark = pytest.mark.slipstream
F64 = torch.float64
SLAB = (0.1, 0.1001)   # 0.1 mm thick: many steps cross it whole
# a round aperture, a central plug (negative radius), slits and one-sided scrapers
KINDS = [("RAD", 1e-3), ("RAD", -1e-3), ("COL_X", 1e-3), ("COL_Y", 1e-3), ("SCR_X", 1e-3), ("SCR_X", -1e-3), ("SCR_Y", 1e-3),
         ("SCR_Y", -1e-3)]


def paths(n=2000, seed=3):
    """Start and end points of steps that cross, graze or miss the slab and the acceptance."""
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(n, 3, dtype=F64, generator=g) * 4e-3 - 2e-3
    b = torch.rand(n, 3, dtype=F64, generator=g) * 4e-3 - 2e-3
    a[:, 2] = SLAB[0] + (torch.rand(n, dtype=F64, generator=g) - 0.7) * 2e-3
    b[:, 2] = a[:, 2] + (torch.rand(n, dtype=F64, generator=g) - 0.3) * 3e-3
    return a, b


def sampled_hit(kind, r, a, b, place, samples):
    s = torch.linspace(0, 1, samples, dtype=F64)[None, :]
    x, y, z = (a[:, i, None] + s * (b[:, i, None] - a[:, i, None]) for i in range(3))
    if place:
        x, z = to_local(x, z, place)
    return ((z >= SLAB[0]) & (z <= SLAB[1]) & _aperture_hit(kind, r, x, y)).any(1)


@pytest.mark.parametrize("place", [None, placement(2e-4, sum(SLAB) / 2, 0.2)], ids=["straight", "placed"])
@pytest.mark.parametrize("kind,r", KINDS)
def test_swept_loss_agrees_with_dense_sampling(kind, r, place):
    a, b = paths()
    aperture = (kind, *SLAB, r) + ((place,) if place else ())
    alive = _apply_apertures(b[:, 0], b[:, 1], b[:, 2], torch.ones(len(a), dtype=torch.bool), (aperture,), prev=(a[:, 0], a[:, 1], a[:, 2]))
    sampled = sampled_hit(kind, r, a, b, place, 2001)
    assert sampled.sum() > 20                      # the sample does exercise this aperture
    assert not (sampled & alive).any()             # target: no loss is missed
    for i in (~sampled & ~alive).nonzero().flatten().tolist():   # grazes between two samples
        assert sampled_hit(kind, r, a[i:i + 1], b[i:i + 1], place, 2_000_001).item(), f"path {i} is lost but never leaves the acceptance"


@pytest.mark.parametrize("kind,r,x,lost", [
    ("RAD", 1e-3, 1e-3, False), ("RAD", 1e-3, math.nextafter(1e-3, 1.0), True),
    ("RAD", -1e-3, 1e-3, False), ("RAD", -1e-3, math.nextafter(1e-3, 0.0), True),
    ("COL_X", 1e-3, -1e-3, False), ("COL_X", 1e-3, -math.nextafter(1e-3, 1.0), True),
])
def test_the_edge_itself_belongs_to_the_acceptance(kind, r, x, lost):
    point = [torch.tensor([v], dtype=F64) for v in (x, 0.0, 0.5)]
    assert _apply_apertures(*point, torch.tensor([True]), ((kind, 0.0, 1.0, r),)).item() is not lost


def test_ring_loses_a_nonfinite_particle():
    ring = RingMap([E("D", "drift", 1.0)], 3e9, device="cpu", dtype=F64)
    state = torch.zeros(2, 6, dtype=F64)
    state[0, 0] = float("nan")
    assert ring.track(state, 1).loss_turn.tolist() == [0, -1]


def test_loss_map_sees_an_excursion_between_turns_that_track_does_not():
    """Documented difference: ``track`` looks at the ring start after each turn, ``track_loss_map`` after every element."""
    ring = RingMap([E("D1", "drift", 1.0), E("K", "hkick", 0.0, kick_x=-2e-3), E("D2", "drift", 1.0)], 3e9, device="cpu", dtype=F64)
    state = torch.zeros(1, 6, dtype=F64)
    state[0, 1] = 1e-3   # 1 mm out at the kick, back near the axis at the end
    end = ring.track(state, 1, aperture=(5e-4, 5e-4))
    inside = ring.track_loss_map(state, 1, aperture=(5e-4, 5e-4))
    assert end.loss_turn.item() == -1
    assert (inside.loss_turn.item(), inside.loss_element.item()) == (0, 0)
