"""IBS bridge for the paper: Slipstream's Bjorken-Mtingwa growth rates on the real ELENA lattice against a live xsuite/xfields calculation on the
same lattice and beam (single-point optics and element-averaged). Runs in WSL (xsuite is not installed on the Windows side):

    SSV_WORKBENCH=<neg_coating_rwi checkout> PYTHONPATH=$SSV_WORKBENCH:$SSV_WORKBENCH/slipstream/src python ibs_elena_bridge.py   # in the xsuite environment

Writes data/2026-10-26_R1/ibs_elena_bridge.json.
"""
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402

from slipstream_dev.examples.elena.validate_ibs import compute_growth_rates, compute_xsuite_growth_rates
from slipstream_dev.examples.elena.validate_ibs_element_averaged import compute_element_averaged_rates

HERE = pathlib.Path(__file__).resolve().parent
OUT = common.EVIDENCE / "2026-10-26_R1" / "ibs_elena_bridge.json"


def row(r):
    return dict(inv_Tx_s=float(r.inv_Tx_s), inv_Ty_s=float(r.inv_Ty_s), inv_Tp_s=float(r.inv_Tp_s))


single, averaged, xs = compute_growth_rates(), compute_element_averaged_rates(), compute_xsuite_growth_rates("bjorken-mtingwa")
gap = lambda a, b: {k: abs(a[k] - b[k]) / abs(b[k]) for k in a}  # noqa: E731
res = dict(single_point=row(single), element_averaged=row(averaged), xsuite=row(xs))
res["gap_single_point"], res["gap_element_averaged"] = gap(res["single_point"], res["xsuite"]), gap(res["element_averaged"], res["xsuite"])
try:
    res["slipstream_commit"] = subprocess.run(["git", "-C", str(common.WORKBENCH / "slipstream"), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
except Exception:  # noqa: BLE001
    pass
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
