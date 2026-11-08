"""Refresh the report's compact, reviewable evidence tables after each campaign phase."""
import argparse
import csv
import json
import pathlib
import subprocess
import sys

import common

HERE = pathlib.Path(__file__).resolve().parent


def fmt(x):
    if x in (None, "", "None"):
        return "--"
    try:
        return f"{float(x):.3g}"
    except (TypeError, ValueError):
        return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--complete", action="store_true", help="mark complete after verifying all planned jobs")
    a = ap.parse_args()
    root = common.RESULTS / a.campaign
    sections, records = [], []
    for case in ("sc_drift", "linear_fodo", "ring_tunes", "ring_grad"):
        summary = root / case / "summary.csv"
        if not summary.exists():
            continue
        rows = list(csv.DictReader(summary.open(newline="")))
        records.append(dict(case=case, rows=rows))
        if case == "sc_drift":
            cols = [("code", "Code"), ("config", "Configuration"), ("warm_s", "Warm s"),
                    ("size_vs_ref", "Size relative error"), ("emit_vs_ref", "Emittance relative error"),
                    ("spread_vs_ref", "Energy-spread relative error"), ("pass", "Size/length/emittance pass")]
        elif case == "ring_tunes":
            cols = [("code", "Code"), ("config", "Configuration"), ("warm_s", "Warm s"),
                    ("dq_ref", "Tune error"), ("dq_prev", "Change on refinement"),
                    ("onturn_x", "First-turn x error m"), ("onturn_px", "First-turn px error"), ("pass", "Tune pass")]
        elif case == "linear_fodo":
            cols = [("code", "Code"), ("config", "Configuration"), ("position_abs_m", "Position error m"),
                    ("momentum_abs", "Momentum error"), ("pass", "Pass")]
        else:
            cols = [("code", "Code"), ("config", "Configuration"), ("fd_s", "FD s"), ("ad_warm_s", "AD s"),
                    ("ad_vs_fd_fro", "AD/self FD norm"), ("rel_fro", "Jacobian/reference norm"), ("status", "Status")]
        title = f"## {case}" + (" (historical reference; see correction below)" if case in ('ring_tunes', 'ring_grad') else "")
        sections += [title, "", "| " + " | ".join(label for _, label in cols) + " |",
                     "| " + " | ".join("---" for _ in cols) + " |"]
        sections += ["| " + " | ".join(fmt(r.get(k)) for k, _ in cols) + " |" for r in rows]
        sections += [""]
    jobs = []
    for p in sorted(root.glob("*/*/*/p*/result.json")):
        r = common.load(p)
        artifacts = {q.name: common.sha256(q) for q in p.parent.glob("*.npz")}
        jobs.append(dict(r, path=p.relative_to(root).as_posix(), result_sha256=common.sha256(p), artifacts_sha256=artifacts))
    evidence = dict(campaign=a.campaign, campaign_complete=a.complete, design=common.load(root / "design.json"), summaries=records, jobs=jobs,
                    source_snapshot_sha256=common.sha256(root / "source_snapshot.zip"),
                    environments={p.parent.name: common.load(p) for p in root.glob("*/environment.json")})
    data = common.EVIDENCE / a.campaign
    data.mkdir(parents=True, exist_ok=True)
    previous = common.load(data / "evidence.json") if (data / "evidence.json").exists() else {}
    if previous.get("campaign_complete") and {(j["path"], j["result_sha256"]) for j in previous["jobs"]} == {(j["path"], j["result_sha256"]) for j in jobs}:
        evidence["campaign_complete"] = True
    if evidence["campaign_complete"] and {r["case"] for r in records} != set(evidence["design"]["cases"]):
        raise ValueError("Cannot mark complete while case summaries are missing")
    if evidence["campaign_complete"]:
        # A completed summary alone does not establish independent repetition.
        for rec in records:
            groups = {}
            for job in (j for j in jobs if j["case"] == rec["case"]):
                key = (job["code"], job["config"]["name"])
                groups.setdefault(key, []).append(job)
            if not groups or any(len(js) != evidence["design"]["procs"] for js in groups.values()):
                raise ValueError("Incomplete independent processes: " + rec["case"])
    (data / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    for rec in records:
        src = root / rec["case"] / "summary.csv"
        (data / f"{rec['case']}.csv").write_bytes(src.read_bytes())
    head = ["# Slipstream benchmark study", "", f"Campaign `{a.campaign}`; " + ("completed tables. " if evidence["campaign_complete"] else "provisional tables; campaign still running. ") +
            f"{sum(j['status'] == 'ok' for j in jobs)}/{len(jobs)} recorded jobs completed successfully.", "",
            "Design: three independent processes and two warm calls per process, one CPU thread; sequential jobs in a seeded random order. "
            "Times include each adapter's diagnostics; executable adapters also include process and file I/O. GPU and CPU results remain labelled. "
            "Warm timing is conditional on an already initialised process. Cold, build, import, memory and process wall time remain in the raw results.", "",
            "Frozen design and source snapshot: `benchmarks/results/" + a.campaign + "/`. "
            "Portable summary and per-job provenance: `data/" + a.campaign + "/evidence.json`. "
            "Reproduce with `python benchmarks/study.py --campaign <new-name>`. No rehearsal outputs are used.", "",
            "References are continuity anchors, not truth: elegant's finest selected ring integrator and ASTRA's 3-D drift. "
            "Passing a tune test does not imply the stricter first-turn map gate passes. "
            "The drift pass gates size, bunch length and emittance only; energy-spread differences are reported separately. "
            "Jacobian norms combine components in their declared numerical SI units and are diagnostic, not unit-invariant physics metrics. "
            "This bounded study does not establish production-scale throughput, CSR fidelity or cathode-emission accuracy.", ""]
    corrected_path = data / 'corrected_reference.json'
    if corrected_path.exists():
        corrected = common.load(corrected_path)
        sections += ['## Corrected momentum reference', '',
                     'Nine new elegant reference jobs use the same normalised reference momentum as the input particles. '
                     'The remaining rows reuse the original saved arrays and timings; no other codes were retimed. '
                     'The original ring tables above and the original Xsuite/CPU control values below are historical. '
                     'Full per-component and gradient results: `data/' + a.campaign + '/corrected_reference.json`. '
                     'Independent bend attribution: `data/' + a.campaign + '/bend_attribution.json`.', '',
                     '| Code | Configuration | Warm s | Tune error | x error m | px error | y error m | py error | Tune pass |',
                     '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        sections += ['| ' + ' | '.join(fmt(r.get(k)) for k in
                     ('code', 'config', 'warm_s', 'dq_ref', 'onturn_x', 'onturn_px', 'onturn_y', 'onturn_py', 'pass')) + ' |'
                     for r in corrected['ring_tunes']]
        sections += ['']
    for filename, title in (("xsuite_model_control.json", "Exploratory Xsuite model correction (historical reference)"),
                            ("gpu_control.json", "Exploratory native GPU Cartesian control"),
                            ("cpu_control.json", "Implemented opt-in CPU compilation (historical reference)"),
                            ("injector_cost_control.json", "Injector diagnostic and refresh presets")):
        path = data / filename
        if not path.exists():
            continue
        supplement = common.load(path)
        sections += ["## " + title, "", "Design and full per-process evidence: `data/" + a.campaign + "/" + filename + "`.", "",
                     "```json", json.dumps(dict(design=supplement["design"], rows=[{k: v for k, v in r.items() if k != "jobs"} for r in supplement["rows"]]), indent=2), "```", ""]
    for filename, title in (('precision_control.json', 'Ring precision control'), ('astra_refinement.json', 'ASTRA 3-D refinement control'),
                            ('injector_step_control.json', 'Injector push-step refinement'), ('native_mesh_control.json', 'Native mesh sensitivity'),
                            ('space_charge_box_probe.json', 'Static Gaussian box and cell-size control'),
                            ('native_box_control.json', 'Native dynamic box-padding control')):
        path = data / filename
        if path.exists():
            supplement = common.load(path)
            sections += ['## ' + title, '', 'Full evidence: `data/' + a.campaign + '/' + filename + '`.', '',
                         '```json', json.dumps(dict(design=supplement.get('design', supplement.get('method')),
                                                    rows=[{k: v for k, v in r.items() if k != 'jobs'}
                                                          for r in supplement['rows']]), indent=2), '```', '']
    (HERE / "BENCHMARK_STUDY.md").write_text("\n".join(head + sections) + "\n")
    print(f"Report refreshed: {len(jobs)} jobs; {len(records)} cases", flush=True)


if __name__ == "__main__":
    main()
