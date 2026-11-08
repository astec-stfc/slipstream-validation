"""Copy the compact evidence of a campaign into the report folder, which reads it for its figures and number checks.

    python -m slipstream_validation.export D:/.../reports/Slipstream            # data/2026-10-26_R1/ <- reviews/claims/evidence/2026-10-26_R1/
    python -m slipstream_validation.export D:/.../reports/Slipstream --campaign 2026-11-20_R2 --dry-run

The validation repository is the master: benchmark and validation runs are made here and the report only reads.  This copies files one
way (never deletes in the report), and says what changed, so the report's git diff shows which numbers moved.
"""
import argparse
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "reviews" / "claims" / "evidence"


def export(report: pathlib.Path, campaign: str, dry: bool = False) -> list:
    src = EVIDENCE / campaign
    if not src.is_dir():
        raise SystemExit(f"no evidence for {campaign} in {EVIDENCE}")
    dst = report / "data" / campaign
    changed = []
    for f in sorted(p for p in src.rglob("*") if p.is_file()):
        rel = f.relative_to(src)
        target = dst / rel
        if target.exists() and target.read_bytes() == f.read_bytes():
            continue
        changed.append(("new" if not target.exists() else "changed", rel.as_posix()))
        if not dry:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
    return changed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", help="the report folder (it has a data/ folder)")
    ap.add_argument("--campaign", default="2026-10-26_R1")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    report = pathlib.Path(a.report)
    if not (report / "data").is_dir():
        sys.exit(f"{report} has no data/ folder")
    changed = export(report, a.campaign, a.dry_run)
    for how, rel in changed:
        print(f"  {how:8} {rel}")
    print(f"{len(changed)} file(s) {'would be ' if a.dry_run else ''}copied into {report / 'data' / a.campaign}")


if __name__ == "__main__":
    main()
