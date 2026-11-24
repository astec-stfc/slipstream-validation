"""First-order tolerance budget of the matched 48 m ring: autograd Jacobian against central differences and Monte Carlo.

The matching control (optimisation_budget_control.py) restores tunes and beta functions with 24 quadrupole knobs.  This control asks what it costs
to learn how those same four outputs (Qx, Qy, beta_x, beta_y at the closed orbit, the outputs of ``joint_optics_loss``) respond to independent random
relative gradient errors on each of the 24 quadrupoles, three ways, on one CPU thread with the eager order-4 map:

    AD   torch.autograd.functional.jacobian of the outputs w.r.t. the relative strength factors (4 reverse sweeps)
    FD   central differences, relative step 1e-5 (2 P forward evaluations)
    MC   N = 40 random machines (one forward evaluation each), the sample standard deviation of each output

The linear prediction sigma_i = sqrt(sum_k J_ik^2 sigma_k^2) is compared with a 400-machine Monte Carlo (not charged to any method) at 1, 3 and 10
times the nominal 0.1 % error, so the curvature of the response is measured rather than assumed.  A knob-count ladder (the first P quadrupoles
free, the rest fixed) times AD and FD as P grows.

    python ring_tolerance_control.py [--out PATH]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from optimisation_budget_control import setup  # noqa: E402

SIGMA = 1e-3              # nominal relative gradient error per quadrupole
SCALES = (1, 3, 10)       # multiples of the nominal error for the linearity check
N_MC, N_REF = 40, 400     # charged Monte Carlo size (as in the injector study) and the uncharged reference
H_FD = 1e-5
LADDER = (1, 2, 5, 12, 24)
REPEATS = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out')
    a = ap.parse_args()
    torch, ring, quads, target, _ = setup()
    from slipstream.analysis import nonlinear as nl
    names = [q['name'] for q in quads]
    k0 = np.array([q['k1'] for q in quads])
    P = len(names)
    beta0 = np.array([float(target.beta_x), float(target.beta_y)])

    def outputs(theta, grad=False):
        """(Qx, Qy, beta_x/beta_x0 - 1, beta_y/beta_y0 - 1) for relative strength factors theta (tensor, length P)."""
        opt = nl.linear_optics(ring.with_strengths(k1={nm: float(k0[i]) * theta[i] for i, nm in enumerate(names)}), create_graph=grad)
        return torch.stack([opt.Qx, opt.Qy, opt.beta_x / float(target.beta_x) - 1, opt.beta_y / float(target.beta_y) - 1])

    def forward(theta):
        with torch.no_grad():
            return outputs(torch.as_tensor(theta, dtype=torch.float64)).numpy()

    def timed(fn, n=REPEATS):
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            r = fn()
            ts.append(time.perf_counter() - t)
        return r, float(np.median(ts))

    def jac_ad(free):
        def f(x):
            th = torch.ones(P, dtype=torch.float64)
            th = th.index_put((torch.as_tensor(free),), x)
            return outputs(th, grad=True)
        return torch.autograd.functional.jacobian(f, torch.ones(len(free), dtype=torch.float64)).numpy()

    def jac_fd(free):
        J = np.zeros((4, len(free)))
        for j, i in enumerate(free):
            up, dn = np.ones(P), np.ones(P)
            up[i] += H_FD
            dn[i] -= H_FD
            J[:, j] = (forward(up) - forward(dn)) / (2 * H_FD)
        return J

    nom = forward(np.ones(P))
    print('nominal Qx, Qy, dbx, dby', nom, flush=True)
    jac_ad(list(range(P)))   # warm up every path before anything is timed
    _, t_fwd = timed(lambda: forward(np.ones(P)))

    ladder = []
    for n in LADDER:
        free = list(range(n))
        _, t_ad = timed(lambda: jac_ad(free))
        _, t_fd = timed(lambda: jac_fd(free))
        ladder.append(dict(P=n, t_ad_s=t_ad, t_fd_s=t_fd, ad_forward_equiv=t_ad / t_fwd, fd_forward_equiv=t_fd / t_fwd))
        print(f"P={n:2d}  AD {t_ad:.3f} s ({t_ad / t_fwd:.1f} fwd)   FD {t_fd:.3f} s ({t_fd / t_fwd:.1f} fwd)", flush=True)

    free = list(range(P))
    J_ad, t_ad = timed(lambda: jac_ad(free))
    J_fd, t_fd = timed(lambda: jac_fd(free))
    rel_fro = float(np.linalg.norm(J_ad - J_fd) / np.linalg.norm(J_fd))
    print(f'AD vs FD Jacobian relative Frobenius error {rel_fro:.2e}', flush=True)

    def sigma_lin(J, s):
        return np.sqrt(((J * s) ** 2).sum(1))

    def mc(n, s, seed):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            out.append(forward(1 + s * rng.standard_normal(P)))
        return np.array(out)

    scales = []
    for m in SCALES:
        s = m * SIGMA
        t = time.perf_counter()
        y40 = mc(N_MC, s, 5000 + m)
        t_mc = time.perf_counter() - t
        ref = mc(N_REF, s, 9000 + m)
        sd_ref, sd40, sd_lin = ref.std(0, ddof=1), y40.std(0, ddof=1), sigma_lin(J_ad, s)
        scales.append(dict(scale=m, sigma_rel=s, sigma_ref=sd_ref.tolist(), sigma_mc40=sd40.tolist(), sigma_lin_ad=sd_lin.tolist(), sigma_lin_fd=sigma_lin(J_fd, s).tolist(),
                           lin_over_ref=(sd_lin / sd_ref).tolist(), mc40_over_ref=(sd40 / sd_ref).tolist(), t_mc40_s=t_mc,
                           mean_shift_over_sigma=((ref.mean(0) - nom) / sd_ref).tolist()))
        print(f"x{m:<2d}  ref sigma {np.array2string(sd_ref, precision=3)}  lin/ref {np.array2string(sd_lin / sd_ref, precision=3)}  mc40/ref {np.array2string(sd40 / sd_ref, precision=3)}  t_mc40 {t_mc:.2f} s", flush=True)

    share = (J_ad * SIGMA) ** 2
    share = share / share.sum(1, keepdims=True)
    top = [[dict(quad=names[k], share=float(share[i, k])) for k in np.argsort(-share[i])[:3]] for i in range(4)]
    out = dict(design=dict(date='2026-11-19', sigma_rel=SIGMA, scales=SCALES, n_mc=N_MC, n_ref=N_REF, h_fd=H_FD, repeats=REPEATS, threads=1, params=P,
                           outputs=['Qx', 'Qy', 'beta_x/beta_x0-1', 'beta_y/beta_y0-1'], objective='eager order-4 RingMap of the 48 m ring, as optimisation_budget_control.py'),
               analysis_sha256=common.sha256(__file__), t_forward_s=t_fwd, t_ad_s=t_ad, t_fd_s=t_fd, t_mc40_nominal_s=scales[0]['t_mc40_s'], jac_ad_vs_fd_rel_frobenius=rel_fro,
               jacobian_ad=J_ad.tolist(), ladder=ladder, scales=scales, top_variance_shares=top)
    path = Path(a.out) if a.out else common.EVIDENCE / '2026-11-19_ring_tolerance/ring_tolerance_control.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + '\n')
    print('wrote', path)


if __name__ == '__main__':
    main()
