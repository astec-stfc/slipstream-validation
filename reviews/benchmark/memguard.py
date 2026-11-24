"""Run a command and kill its whole process tree when it, or the machine, runs short of memory.

    python memguard.py --max-gib 6 --reserve-gib 12 -- python -u injector_clara_profile.py --image on --json out.json

The job is killed (exit code 99) when the tree's working set exceeds ``--max-gib`` or the free physical memory of the machine falls below ``--reserve-gib``.
The peak working set is printed when the job ends, so a later campaign can size how many jobs run at once. On the PC of the paper the pagefile is on a hard disk:
a job that runs the machine out of memory makes it unusable (21 November 2026), so every long CPU job of a campaign should run under this guard.
"""
import argparse
import subprocess
import sys
import time

import psutil


def tree_rss(proc):
    try:
        procs = [proc] + proc.children(recursive=True)
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for p in procs:
        try:
            total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--max-gib', type=float, required=True, help='kill the job when its process tree holds more than this')
    ap.add_argument('--reserve-gib', type=float, default=12.0, help='kill the job when the free physical memory of the machine is below this')
    ap.add_argument('--interval', type=float, default=1.0)
    ap.add_argument('cmd', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == '--' else a.cmd
    job = subprocess.Popen(cmd)
    ps, peak, why = psutil.Process(job.pid), 0, None
    while job.poll() is None:
        rss, free = tree_rss(ps), psutil.virtual_memory().available
        peak = max(peak, rss)
        if rss > a.max_gib * 2**30:
            why = f'its working set {rss / 2**30:.1f} GiB exceeds {a.max_gib} GiB'
        elif free < a.reserve_gib * 2**30:
            why = f'free memory {free / 2**30:.1f} GiB is below the reserve of {a.reserve_gib} GiB'
        if why:
            for p in ps.children(recursive=True) + [ps]:
                try:
                    p.kill()
                except psutil.NoSuchProcess:
                    pass
            print(f'memguard: killed {cmd[0]}: {why}', file=sys.stderr, flush=True)
            break
        time.sleep(a.interval)
    job.wait()
    print(f'memguard: peak working set {peak / 2**30:.2f} GiB', file=sys.stderr, flush=True)
    sys.exit(99 if why else job.returncode)


if __name__ == '__main__':
    main()
