"""Collect the injector sensitivity profile and the reverse-mode memory measurements into data/2026-10-26_R1/.

Inputs (written by slipstream_dev/benchmark/injector_ad_profile.py and injector_ad_memory.py, 3 November 2026, CLARA 5 pC deck):
  results/2026-11-03_injector_ad_profile.json  -> data/2026-10-26_R1/injector_ad_profile.json
  results/2026-11-03_injector_ad_memory.json   -> data/2026-10-26_R1/injector_ad_memory.json (with the GPU's total memory)
"""
import json
from pathlib import Path

import common
import torch

HERE = Path(__file__).resolve().parent
DATA = common.EVIDENCE / '2026-10-26_R1'


def main():
    for name in ('injector_ad_profile', 'injector_ad_memory'):
        src = common.RESULTS / f'2026-11-03_{name}.json'
        d = json.loads(src.read_text())
        d['source_sha256'] = {src.name: common.sha256(src)}
        d['analysis_sha256'] = common.sha256(__file__)
        if name == 'injector_ad_memory':
            d['device_MiB'] = torch.cuda.get_device_properties(0).total_memory / 2**20 if torch.cuda.is_available() else 8 * 1024.0
        (DATA / f'{name}.json').write_text(json.dumps(d, indent=1) + '\n')
        print(name, 'written')


if __name__ == '__main__':
    main()
