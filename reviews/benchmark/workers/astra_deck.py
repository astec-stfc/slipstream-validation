"""ASTRA input for the ``sc_drift`` case, shared by the ASTRA adapter and the Slipstream injector adapter (which reads the deck
with ``Injector.from_astra``).  Numpy only."""
import pathlib

import numpy as np

ASTRA_MC2_EV = 511003.0  # ASTRA's electron rest energy, not CODATA


def write_deck(d, lab, case, nrad=32, nlong=64, h_ns=0.0035, three_d=False, n3d=32, cell_var=2, name="sc_drift", mc2_eV=ASTRA_MC2_EV):
    """Write ``<name>.in`` and ``beam.ini`` into directory ``d``.  ``lab`` is the (N, 6) case array x, y, z, bgx, bgy, bgz.
    Momenta are written as ``mc2_eV * beta*gamma``: ASTRA divides by its own 511003.0 eV (the default); a code that divides by the CODATA mass
    (Slipstream's deck reader) needs ``mc2_eV=case['mass_eV']`` to receive the same beta*gamma."""
    d = pathlib.Path(d)
    n = len(lab)
    x, y, z, bgx, bgy, bgz = lab.T
    p = mc2_eV * np.column_stack([bgx, bgy, bgz])  # eV/c
    pref = mc2_eV * bgz.mean()
    q_nC = case["charge_C"] * 1e9 / n  # negative for electrons
    rows = np.zeros((n + 1, 10))
    rows[0] = [0, 0, 0, 0, 0, pref, 0, q_nC, 1, 5]  # reference particle, absolute
    rows[1:, 0], rows[1:, 1], rows[1:, 2] = x, y, z
    rows[1:, 3], rows[1:, 4], rows[1:, 5] = p[:, 0], p[:, 1], p[:, 2] - pref  # pz relative to the reference
    rows[1:, 7], rows[1:, 8], rows[1:, 9] = q_nC, 1, 5
    np.savetxt(d / "beam.ini", rows, fmt="%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %d %d")
    stations = case["stations_m"]
    zstop = stations[-1]
    sc3 = f"LSPCH3D = T, Nxf = {n3d}, Nyf = {n3d}, Nzf = {n3d}," if three_d else "LSPCH3D = F, Nrad = %d, Nlong_in = %d, Cell_var = %d, min_grid = 3.424657e-13, Max_scale = 0.1, Max_count = 100," % (nrad, nlong, cell_var)
    (d / f"{name}.in").write_text(f"""&NEWRUN
 Head = 'sc_drift', RUN = 1, Loop = F, Lprompt = F, Auto_phase = F, Track_All = T, Distribution = 'beam.ini',
 H_max = {h_ns!r}, H_min = {h_ns!r}, High_res = T,
/
&OUTPUT
 ZSTART = 0, ZSTOP = {zstop}, Lmagnetized = F, Zemit = {int(round(zstop / 0.05))}, Zphase = 1, RefS = T, EmitS = T, PhaseS = T, TrackS = F, LandFS = F,
/
&CHARGE
 LSPCH = T, Lmirror = F, {sc3}
/
""")
    return d / f"{name}.in"
