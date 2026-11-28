"""Space charge of electrons at rest on the cathode: a thin uniform cylinder against its closed form, the hardest moment of emission.

A cylinder of radius R and length L with uniform charge density rho sits on the cathode (z = 0).  With its image, the axial field at
the centre of the cathode is E(0) = rho / eps0 * (sqrt(L^2 + R^2) - R - L), and the transverse field vanishes everywhere on the plane.
The sample is a Sobol sequence, so the numbers repeat from host to host.
"""
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("slipstream")

from slipstream.constants import EPS0_F_PER_M  # noqa: E402
from slipstream.tracking.native.injector import SpaceChargeSettings, _compute_space_charge  # noqa: E402

pytestmark = pytest.mark.slipstream
F64 = torch.float64
CHARGE_C, RADIUS_M, N = -100e-12, 1e-3, 200_000
PROBES = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.0, 0.5)]   # (x, y) / R on the cathode; the first is the centre


def surface_field(length_m, **settings):
    """(relative error of Ez at the centre, largest transverse field on the cathode / exact Ez) from probes of zero charge."""
    u = torch.quasirandom.SobolEngine(3, scramble=True, seed=1).draw(N, dtype=F64)
    r, phi = RADIUS_M * u[:, 0].sqrt(), 2 * math.pi * u[:, 1]
    p = torch.tensor(PROBES, dtype=F64) * RADIUS_M
    x = torch.cat([r * torch.cos(phi), p[:, 0]])
    y = torch.cat([r * torch.sin(phi), p[:, 1]])
    z = torch.cat([length_m * u[:, 2], torch.zeros(len(p), dtype=F64)])
    q = torch.cat([torch.full((N,), CHARGE_C / N, dtype=F64), torch.zeros(len(p), dtype=F64)])
    rest = torch.zeros_like(x)
    sc = SpaceChargeSettings(cathode_image=True, **settings)
    # private, as in Slipstream's own tests: the dispatch that Injector.track calls at every field update
    Ex, Ey, Ez, *_ = _compute_space_charge(x, y, z, q, rest, rest, rest, torch.ones_like(x, dtype=torch.bool), "cpu", F64, {}, sc, 0.0,
                                           emitted_all=True)
    rho = CHARGE_C / (math.pi * RADIUS_M**2 * length_m)
    exact = rho / EPS0_F_PER_M * (math.hypot(length_m, RADIUS_M) - RADIUS_M - length_m)
    on_plane = slice(-len(p), None)
    transverse = max(Ex[on_plane].abs().max().item(), Ey[on_plane].abs().max().item())
    return Ez[-len(p)].item() / exact - 1, transverse / abs(exact)


@pytest.mark.parametrize("geometry", ["cylindrical", "3d"])
@pytest.mark.parametrize("length_m", [1e-6, 1e-4, 1e-3])
def test_cathode_is_an_equipotential(geometry, length_m):
    assert surface_field(length_m, geometry=geometry)[1] < 1e-12   # target: exact by the image construction


@pytest.mark.parametrize("length_m,limit", [(1e-6, 0.05), (1e-5, 0.05), (5e-5, 0.05), (2e-4, 0.05), (1e-3, 0.08)])
def test_cartesian_solver_surface_field_at_any_slab_thickness(length_m, limit):
    """The 3-D box follows the particles, so the slab is resolved however thin.  Ratchet: measured -2.3 % to -3.9 %."""
    assert abs(surface_field(length_m, geometry="3d")[0]) < limit


@pytest.mark.parametrize("length_m", [1e-6, 1e-3])
def test_cartesian_solver_surface_field_converges(length_m):
    coarse, fine = (abs(surface_field(length_m, geometry="3d", n_xyz=[n] * 3)[0]) for n in (32, 64))
    assert fine < 0.7 * coarse


# Ratchets of the default cylindrical mesh, about twice the value measured on 28 November 2026.  The error is not monotonic in L: the
# mesh cannot be finer along z than sigma_z_floor_m = 50 um allows.
@pytest.mark.parametrize("length_m,limit", [(1e-5, 0.03), (5e-5, 0.27), (2e-4, 0.15), (1e-3, 0.01)])
def test_cylindrical_solver_surface_field(length_m, limit):
    assert abs(surface_field(length_m)[0]) < limit


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "resolution limit of the default mesh: sigma_z_floor_m = 50 um gives cells of about 4 um along z, and a 1 um slab reads 85 % low "
    "at the cathode.  The 3-D solver is within 3 %.  Lowering the floor alone does not cure it: see the next test."))
def test_cylindrical_solver_resolves_a_slab_thinner_than_its_cells():
    assert abs(surface_field(1e-6)[0]) < 0.15


def test_image_is_dropped_when_the_adaptive_mesh_would_exceed_its_cap():
    """Documented behaviour (SpaceChargeSettings, "the cathode image auto-disables"), pinned because nothing says so at run time:
    with a floor small enough to resolve a 1 um slab, the plane is no longer an equipotential."""
    assert surface_field(1e-6, sigma_z_floor_m=5e-7)[1] > 0.1
