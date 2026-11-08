"""Run a command and record the peak summed resident set of its whole process tree from /proc (stdlib only; for WSL/Linux).

    python wsl_tree_rss.py --out result.json -- <python args...>

The child is ``sys.executable <python args...>``.  Writes {tree_peak_bytes, own_peak_bytes, max_processes, wall_s, returncode}; sampled every 25 ms.
"""
import json
import os
import subprocess
import sys
import time

PAGE = os.sysconf("SC_PAGE_SIZE")
INTERVAL = 0.025


def snapshot():
    rss, parent = {}, {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat") as f:
                    stat = f.read()
                with open(f"/proc/{d}/statm") as f:
                    pages = int(f.read().split()[1])
            except OSError:
                continue
            parent[int(d)] = int(stat.rsplit(")", 1)[1].split()[1])
            rss[int(d)] = pages * PAGE
    return rss, parent


def tree(root, rss, parent):
    kids = {}
    for pid, ppid in parent.items():
        kids.setdefault(ppid, []).append(pid)
    stack, members = [root], []
    while stack:
        p = stack.pop()
        if p in rss:
            members.append(p)
        stack.extend(kids.get(p, []))
    return sum(rss[p] for p in members), rss.get(root, 0), len(members)


def main():
    a = sys.argv[1:]
    out, cmd = a[a.index("--out") + 1], a[a.index("--") + 1:]
    t0 = time.perf_counter()
    proc = subprocess.Popen([sys.executable, *cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    peak = dict(tree_peak_bytes=0, own_peak_bytes=0, max_processes=0)
    while proc.poll() is None:
        t, o, n = tree(proc.pid, *snapshot())
        peak["tree_peak_bytes"], peak["own_peak_bytes"], peak["max_processes"] = max(peak["tree_peak_bytes"], t), max(peak["own_peak_bytes"], o), max(peak["max_processes"], n)
        time.sleep(INTERVAL)
    peak.update(wall_s=time.perf_counter() - t0, returncode=proc.returncode)
    with open(out, "w") as f:
        json.dump(peak, f)


if __name__ == "__main__":
    main()
