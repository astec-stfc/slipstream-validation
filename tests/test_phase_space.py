"""Tier 0: the distribution metrics see what rms sizes and emittances do not."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "reviews" / "benchmark"))
import phase_space  # noqa: E402

from slipstream_validation import check  # noqa: E402

pytestmark = pytest.mark.tier0
N = 100_000


def gaussian(seed, n=N):
    return np.random.default_rng(seed).standard_normal((n, 2))


def test_whitening_removes_mean_size_and_correlation():
    a = gaussian(1)
    sheared = a @ np.array([[1e-3, 0.0], [4e-4, 2e-5]]) + [5.0, -2.0]
    w = phase_space.whiten(sheared)
    np.testing.assert_allclose(w.mean(0), 0, atol=1e-9)
    np.testing.assert_allclose(np.cov(w, rowvar=False, bias=True), np.eye(2), atol=1e-9)


def test_sliced_wasserstein_has_its_known_values():
    a = gaussian(1)
    assert phase_space.sliced_wasserstein(a, a) == 0
    # a shift d projects to |d cos(angle)|, whose mean over all angles is 2 |d| / pi
    assert phase_space.sliced_wasserstein(a, a + [0.1, 0.0]) == pytest.approx(0.2 / np.pi, rel=1e-3)
    assert phase_space.sliced_wasserstein(a, gaussian(2, N // 2)) < 0.01   # unequal numbers: only sampling noise


def test_a_halo_with_the_same_rms_is_caught():
    core, other = gaussian(1), gaussian(2)
    halo = gaussian(3)
    halo[: N // 100] *= 6.0                      # 1 % of the particles in a wide halo
    halo /= halo.std(0)                          # the rms size is that of the core again
    np.testing.assert_allclose(halo.std(0), core.std(0), rtol=0.01)
    same = phase_space.sliced_wasserstein(phase_space.whiten(core), phase_space.whiten(other))
    seen = phase_space.sliced_wasserstein(phase_space.whiten(core), phase_space.whiten(halo))
    assert seen > 10 * same
    radius = lambda a: np.quantile(np.hypot(*phase_space.whiten(a).T), 0.999)  # noqa: E731
    assert abs(radius(halo) / radius(core) - 1) > 0.2 > 10 * abs(radius(other) / radius(core) - 1)


def bunch(seed, ct=0.0):
    rng = np.random.default_rng(seed)
    x, y, z = (rng.standard_normal(20_000) * s for s in (1e-3, 1e-3, 3e-4))
    bgx, bgy = (rng.standard_normal(20_000) * 2e-3 for _ in range(2))
    bgz = 10.8 + rng.standard_normal(20_000) * 1e-2
    gamma = np.sqrt(1 + bgx**2 + bgy**2 + bgz**2)
    return dict(x=x + bgx / gamma * ct, y=y + bgy / gamma * ct, z=3.0 + z + bgz / gamma * ct, bgx=bgx, bgy=bgy, bgz=bgz)


def test_snapshots_at_different_times_compare_equal():
    found = phase_space.compare(bunch(1, ct=0.05), bunch(1), 3.0)
    assert max(found.values()) < 1e-3, found


def test_compare_reports_the_worse_transverse_plane():
    odd = bunch(1)
    odd["y"][:200] *= 8
    found = phase_space.compare(odd, bunch(1), 3.0)
    assert found["shape"] > 0.02 and found["halo"] > 0.2 and found["shape_z"] < 1e-3, found


def test_the_distribution_gate_can_fail():
    row = dict(status="ok", size_vs_ref="0.01", length_vs_ref="0.01", emit_vs_ref="0.01", spread_vs_ref="0.01", dEkin_change_eV="50",
               shape_vs_ref="0.004", halo_vs_ref="0.004", shape_z_vs_ref="0.004", halo_z_vs_ref="0.004", **{"pass": "True"})

    def judge(**change):   # the archived row has no distribution columns
        old = {k: v for k, v in row.items() if not k.startswith(("shape", "halo"))}
        return check.judge("sc_drift", "slipstream", "c", dict(row, **change), old, {})[0]

    assert judge() == "ok"
    assert judge(halo_vs_ref="0.05") == "LIMIT"
    assert judge(shape_vs_ref="missing") == "LIMIT"
    assert judge(shape_vs_ref="") == "ok"            # a code whose worker writes no snapshot
