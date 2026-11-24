"""Collect the injector sensitivity profile, the energy-spread check and the reverse-mode memory measurements into reviews/claims/evidence/TAG/.

Archived campaign (no argument; written by slipstream_dev/benchmark/injector_ad_profile.py and injector_ad_memory.py, 3 November 2026, CLARA 5 pC deck, cathode image off):
  results/2026-11-03_injector_ad_profile.json  -> evidence/2026-10-26_R1/injector_ad_profile.json
  results/2026-11-03_injector_ad_memory.json   -> evidence/2026-10-26_R1/injector_ad_memory.json (with the GPU's total memory)

``--campaign TAG`` (written by injector_clara_profile.py, injector_clara_energy_spread.py and injector_clara_memory.py; the profile is one file per group of stations):
  results/TAG/profile/part*.json (rows merged, sorted by station)  -> evidence/TAG/injector_ad_profile.json
  results/TAG/energy_spread.json                                   -> evidence/TAG/injector_fd_energy_spread.json
  results/TAG/memory/injector_ad_memory.json                       -> evidence/TAG/injector_ad_memory.json (with the GPU's total memory)
"""
import argparse
import json
from pathlib import Path

import common
import torch

HERE = Path(__file__).resolve().parent


def _jobs(campaign):
    """(output name, [source files], loader) for each file of the evidence."""
    if not campaign:
        return [(n, [common.RESULTS / f'2026-11-03_{n}.json']) for n in ('injector_ad_profile', 'injector_ad_memory')]
    res = common.RESULTS / campaign
    return [('injector_ad_profile', sorted((res / 'profile').glob('part*.json'))), ('injector_fd_energy_spread', [res / 'energy_spread.json']),
            ('injector_ad_memory', [res / 'memory' / 'injector_ad_memory.json'])]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--campaign', help='a new campaign instead of the archived 2026-10-26_R1 one')
    a = ap.parse_args()
    tag = a.campaign or '2026-10-26_R1'
    (common.EVIDENCE / tag).mkdir(parents=True, exist_ok=True)
    for name, srcs in _jobs(a.campaign):
        d = json.loads(srcs[0].read_text())
        if len(srcs) > 1:
            d['rows'] = sorted((r for s in srcs for r in json.loads(s.read_text())['rows']), key=lambda r: r['z_stop_m'])
        d['source_sha256'] = {s.name: common.sha256(s) for s in srcs}
        d['analysis_sha256'] = common.sha256(__file__)
        if name == 'injector_ad_memory':
            d['device_MiB'] = torch.cuda.get_device_properties(0).total_memory / 2**20 if torch.cuda.is_available() else 8 * 1024.0
        (common.EVIDENCE / tag / f'{name}.json').write_text(json.dumps(d, indent=1) + '\n')
        print(name, 'written')


if __name__ == '__main__':
    main()
