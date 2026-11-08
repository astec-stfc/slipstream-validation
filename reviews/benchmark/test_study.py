"""Independent numerical controls for the benchmark harness (no external programs)."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common
import run
from analyse_grad import best_fd
from analyse_transport import reference
from analyse_sc import solver_family, energy_gaps
from study import DESIGN


def test_finite_energy_coordinate_roundtrip():
    delta = np.array([-0.5, -1e-6, 0, 1e-6, 0.5])
    for energy in (0.6e6, 5.51099895e6, 629e6):
        p = common.delta_to_cheetah_p(delta, energy)
        np.testing.assert_allclose(common.cheetah_p_to_delta(p, energy), delta, rtol=3e-15, atol=1e-17)
        pc = common.p0c_eV(energy)
        # Independently calculate the energy of a finite momentum offset.
        np.testing.assert_allclose(energy + pc * p, np.sqrt((pc * (1 + delta))**2 + common.MC2_EV**2), rtol=2e-15)


def test_canonical_slope_roundtrip():
    xp, yp, delta = np.array([0, .4, -.3]), np.array([.2, -.1, .5]), np.array([-.2, .1, .3])
    px, py = common.slopes_to_canon(xp, yp, delta)
    sx, sy = common.canon_to_slopes(px, py, delta)
    np.testing.assert_allclose(sx, xp, atol=2e-16)
    np.testing.assert_allclose(sy, yp, atol=2e-16)


def test_tune_estimator_exact_rotation():
    t = np.arange(1024)
    for q in (.14321, .27345):
        got, _ = common.naff(np.exp(2j * np.pi * q * t), q, .04)
        assert abs(got - q) < 1e-9


def test_analytic_fodo_preserves_transverse_symplectic_form():
    lat = common.load(common.ROOT / "cases" / "linear_fodo_lat.json")
    M = reference(lat)
    S = np.kron(np.eye(2), [[0, 1], [-1, 0]])
    np.testing.assert_allclose(M.T @ S @ M, S, atol=2e-15)
    np.testing.assert_allclose(reference(dict(elements=[dict(kind="drift", length=2)])),
                               [[1, 2, 0, 0], [0, 1, 0, 0], [0, 0, 1, 2], [0, 0, 0, 1]])


def test_fd_extrapolation_recovers_known_derivative():
    steps = np.array([1e-2, 1e-3, 1e-4, 1e-5])
    exact = np.array([[1, 2], [3, 4.]])
    got, _ = best_fd(np.array([exact + h*h * 7 for h in steps]), steps)
    np.testing.assert_allclose(got, exact, atol=1e-14)


def test_every_listed_code_has_a_selected_case():
    assert set(run.REG) - {c for d in DESIGN.values() for c in d["codes"].split(",")} == set()   # every code of bench.toml is in the frozen design
    for case, d in DESIGN.items():
        for code in d["codes"].split(","):
            assert run.worker_path(code, case)


def test_all_case_inputs_are_finite():
    for case in DESIGN:
        inputs = common.launches(common.load(common.ROOT / "cases" / f"{case}.json"))
        assert inputs.ndim == 2 and inputs.shape[1] == 6
        assert np.isfinite(inputs).all()


def test_solver_changes_are_not_mesh_refinement():
    assert solver_family(dict(geometry="cylindrical")) != solver_family(dict(geometry="3d"))
    assert solver_family(dict(space_charge="Gauss3D")) != solver_family(dict(space_charge="3D"))
    assert solver_family(dict(geometry="3d", grid=32)) == solver_family(dict(geometry="3d", grid=64))


def test_missing_ring_reference_cannot_be_replaced_by_consensus(tmp_path, monkeypatch):
    import analyse
    (tmp_path / "cases").mkdir()
    (tmp_path / "cases" / "ring_tunes.json").write_text((common.ROOT / "cases" / "ring_tunes.json").read_text())
    monkeypatch.setattr(analyse, "HERE", tmp_path)
    monkeypatch.setattr(sys, "argv", ["analyse.py", "--campaign", "empty", "--case", "ring_tunes"])
    with pytest.raises(RuntimeError, match="No successful declared reference: elegant"):
        analyse.main()


def test_constant_energy_offset_is_separate_from_force_error():
    ref = np.array([[5e6], [5e6 + 1], [5e6 + 2]])
    offset = ref + 2400
    gap = energy_gaps(offset, ref, ["ekin_eV"])
    assert gap == dict(dEkin_eV=2400, baseline_dEkin_eV=2400, dEkin_change_eV=0)
    offset[:, 0] += [0, .5, 1]
    assert energy_gaps(offset, ref, ["ekin_eV"])["dEkin_change_eV"] == 1


def test_nonfinite_ad_is_rejected_before_being_published(tmp_path):
    lat = dict(elements=[dict(name="Q", k1=2.)])
    case = dict(knobs=[dict(family="Q", kind="k1")], fd_rel_steps=[.01, .001])
    def build(lat, case, cfg):
        return lat["elements"][0]["k1"], None

    def track(model, init, turns, cfg):
        state = np.zeros((1, turns, 4))
        state[0, -1, 0] = model
        return state

    ad = (lambda *args: None, lambda *args: (np.zeros(4), np.full((4, 1), np.nan)))
    with pytest.raises(ValueError, match="Non-finite gradient output: J_ad"):
        common.grad_flow(case, lat, {}, build, track, ad, np.zeros((1, 6)), 2, lambda dev: None,
                         "cpu", tmp_path, 1)
    assert not (tmp_path / "grad.npz").exists()


def test_ocelot_zero_charge_matches_ballistic_drift():
    pytest.importorskip("ocelot")
    sys.path.insert(0, str(common.ROOT / "workers"))
    import w_ocelot_sc as w
    case = common.load(common.ROOT / "cases" / "sc_drift.json")
    case = dict(case, charge_C=0.0, stations_m=[0.0, 0.1, 0.2])
    cfg = w.CONFIGS[0]
    model, _ = w.build({}, case, cfg)
    got = w.track(model, model["lab"], 1, cfg)
    lab = model["lab"]
    pos, bg = lab[:, :3].copy(), lab[:, 3:]
    gamma = np.sqrt(1 + (bg * bg).sum(axis=1))
    gamma0 = case["energy_eV"] / case["mass_eV"]
    beta0 = np.sqrt(1 - gamma0**-2)
    ct = case["stations_m"][-1] / beta0
    pos += bg / gamma[:, None] * ct
    pos[:, 2] -= case["stations_m"][-1]
    truth = common.lab_stats(*pos.T, *bg.T, mass_eV=case["mass_eV"])
    columns = ("sigma_x_m", "sigma_y_m", "sigma_z_m", "eps_nx_m", "eps_ny_m", "ekin_eV", "sigma_E_eV")
    np.testing.assert_allclose(got[-1, 1:], [truth[k] for k in columns], rtol=2e-5, atol=1e-12)


def test_elegant_reference_momentum_matches_particle_normalisation(tmp_path, monkeypatch):
    sys.path.insert(0, str(common.ROOT / "workers"))
    import w_elegant
    monkeypatch.setattr(w_elegant.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(w_elegant.subprocess, "run", lambda *args, **kwargs: None)
    case = common.load(common.ROOT / "cases/ring_tunes.json")
    lat = common.load(common.ROOT / "cases" / case["lattice"])
    model, _ = w_elegant.build(lat, case, w_elegant.CONFIGS[-1])
    setup = (tmp_path / "run.ele").read_text()
    pbg = model["p0c"] / model["m"]
    assert f"p_central={pbg!r}" in setup
    assert "p_central_mev=" not in setup
    assert np.allclose(np.loadtxt(tmp_path / "beam.txt")[:, 5], pbg)


@pytest.mark.parametrize('double_single', [False, True])
def test_adapter_uses_input_precision_for_mode(double_single):
    import torch
    from types import SimpleNamespace
    sys.path.insert(0, str(common.ROOT / "workers"))
    import w_slipstream

    class Ring:
        device, dtype = "cpu", torch.float32

        def __init__(self):
            self.double_single = double_single

        def track(self, state, turns, **kwargs):
            assert state.dtype == (torch.float64 if double_single else torch.float32)
            return SimpleNamespace(tbt=state[:, :4].unsqueeze(0))

    initial = np.array([[0.0010000000001, 0, 0, 0, 0, 0]])
    result = w_slipstream.track(Ring(), initial, 1, {})
    assert result[0, 0, 0] == initial[0, 0]


def test_astra_refinement_changes_one_control_at_a_time():
    sys.path.insert(0, str(common.ROOT / "workers"))
    import w_astra
    configs = {c['name']: c for c in w_astra.CONFIGS}
    base = configs['cpu-f64-3d32h1.4']
    grid = configs['cpu-f64-3d64h1.4']
    step = configs['cpu-f64-3d32h0.7']
    assert grid['three_d'] and grid['n3d'] == 2 * base['n3d'] and grid['h_ns'] == base['h_ns']
    assert step['three_d'] and step['n3d'] == base['n3d'] and step['h_ns'] == base['h_ns'] / 2


def test_double_single_jacobian_is_explicitly_unsupported():
    sys.path.insert(0, str(common.ROOT / "workers"))
    import w_slipstream
    with pytest.raises(NotImplementedError, match='Jacobian adapter'):
        w_slipstream.build({}, {'kind': 'grad'}, {'double_single': 'hybrid'})


def test_injector_step_refinement_preserves_refresh_and_record_intervals():
    sys.path.insert(0, str(common.ROOT / "workers"))
    import w_slipstream_sc
    configs = {c['name']: c for c in w_slipstream_sc.CONFIGS}
    base = configs['cuda-f32-3d32t1c10-r100']
    fine = configs['cuda-f32-3d32t0.5c20-r200']
    assert fine['dt_ps'] == base['dt_ps'] / 2
    assert fine['dt_ps'] * fine['subcycle'] == base['dt_ps'] * base['subcycle']
    assert fine['dt_ps'] * fine['record_every'] == base['dt_ps'] * base['record_every']
    assert fine['geometry'] == base['geometry'] and fine['dtype'] == base['dtype']


def test_injector_mesh_refinement_preserves_other_controls(monkeypatch, tmp_path):
    sys.path.insert(0, str(common.ROOT / "workers"))
    import w_slipstream_sc
    from types import SimpleNamespace
    from slipstream.tracking.native.injector import Injector, SpaceChargeSettings

    configs = {c['name']: c for c in w_slipstream_sc.CONFIGS}
    base = configs['cuda-f32-3d32t1c10-r100']
    monkeypatch.setattr(w_slipstream_sc.tempfile, 'mkdtemp', lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(w_slipstream_sc, 'write_deck', lambda *args: tmp_path / 'fake.in')
    monkeypatch.setattr(w_slipstream_sc, 'launches', lambda case: np.zeros((1, 6)))
    monkeypatch.setattr(Injector, 'from_astra', lambda *args: SimpleNamespace(space_charge=SpaceChargeSettings()))
    for n in (16, 64):
        other = configs[f'cuda-f32-3d{n}t1c10-r100']
        assert other['n3d'] == n
        assert all(other[k] == base[k] for k in ('device', 'dtype', 'dt_ps', 'subcycle', 'record_every', 'geometry'))
        model, _ = w_slipstream_sc.build({}, {'stations_m': [0, 3]}, other)
        assert model['inj'].space_charge.n_xyz == [n] * 3
    for n, padding in ((64, .05), (64, .2), (81, .2)):
        config = configs[f'cuda-f32-3d{n}t1c10-r100-p{padding:g}']
        model, _ = w_slipstream_sc.build({}, {'stations_m': [0, 3]}, config)
        assert model['inj'].space_charge.n_xyz == [n] * 3
        assert model['inj'].space_charge.box_padding_3d == padding


def test_deck_momentum_follows_the_requested_mass(tmp_path):
    from astra_deck import ASTRA_MC2_EV, write_deck
    case = dict(charge_C=-20e-12, stations_m=[0.0, 0.1])
    lab = np.array([[1e-3, 0, 0, 0, 0, 10.8], [-1e-3, 0, 0, 0, 0, 10.8]])
    write_deck(tmp_path, lab, case)
    write_deck(tmp_path / '..', lab, case, mc2_eV=common.MC2_EV)
    default = np.loadtxt(tmp_path / 'beam.ini')[0, 5]
    fixed = np.loadtxt(tmp_path / '..' / 'beam.ini')[0, 5]
    np.testing.assert_allclose(default / fixed, ASTRA_MC2_EV / common.MC2_EV, rtol=1e-15)


def test_mass_convention_offset_matches_the_measured_43_ev():
    from charge_off_control import mass_offset_eV
    assert abs(mass_offset_eV(5e6, 510998.95) - 43.30) < 0.01


def test_gradient_acceptance_decision_is_unit_invariant():
    import gradient_acceptance as ga
    rng = np.random.default_rng(1)
    k0, sigma = rng.uniform(1, 3, 5), rng.uniform(1e-4, 1e-2, 4)
    ref, J = rng.normal(size=(4, 5)), None
    J = ref * (1 + 1e-4 * rng.normal(size=(4, 5)))
    base = ga.metrics(ga.scaled(J, k0, sigma), ga.scaled(ref, k0, sigma))
    for units in ga.UNIT_VARIANTS.values():
        u = np.asarray(units)
        again = ga.metrics(ga.scaled(J * u[:, None], k0, sigma * u), ga.scaled(ref * u[:, None], k0, sigma * u))
        assert again['passed'] == base['passed']
        assert abs(again['e_abs'] - base['e_abs']) < 1e-12 and abs(again['e_rel'] - base['e_rel']) < 1e-12


def test_richardson_solve3_recovers_limit_and_order_and_rejects_nonmonotone():
    import richardson_reference as rr
    n = (32, 64, 128)
    limit, order = rr.solve3(n, [1.0 + 2.0 * m ** -1.7 for m in n])
    assert abs(limit - 1.0) < 1e-9 and abs(order - 1.7) < 1e-9
    assert rr.solve3(n, [1.0, 1.5, 1.2]) is None      # oscillating ladder: no extrapolation
    assert rr.solve3(n, [1.0, 1.5, 2.2]) is None      # diverging ladder


def test_unit_spread_of_a_zero_metric_is_one():
    import gradient_acceptance as ga
    assert ga.ratio([0.0, 0.0]) == 1.0 and ga.ratio([1.0, 2.0]) == 2.0


def test_longitudinal_gate_thresholds_are_declared_and_retrospective_in_the_docstring():
    import longitudinal_gates as lg
    assert lg.SPREAD_GATE == 0.05 and lg.GAIN_GATE_EV == 5.0
    assert 'RETROSPECTIVE' in lg.__doc__
