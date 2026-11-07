"""Persistent-job wrapper for the IJIS E1/E2 runs (repository persistent-job policy).

Usage (launched detached, for example with PowerShell Start-Process):
    python job_wrapper.py <job_key> -- <python> <script> <args...>

Writes under OUT/jobs/<job_key>/: state.json, pid (wrapper and child), stdout.log, stderr.log,
exit_code and a DONE marker. Refuses to start when the same job key already has a DONE marker, a
live recorded PID, or a live OS process whose command line names the job key (one writer only).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

OUT = Path(__file__).resolve().parents[1]
JOBS = OUT / "jobs"


def write_state(job_dir: Path, state: dict) -> None:
    tmp = job_dir / "state.json.new"
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(job_dir / "state.json")


def other_live_process(job_key: str) -> list[int]:
    me = os.getpid()
    parent = os.getppid()
    hits = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = p.info.get("cmdline") or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if p.info["pid"] in (me, parent):
            continue
        if any("job_wrapper.py" in c for c in cmd) and job_key in cmd:
            hits.append(p.info["pid"])
    return hits


def main() -> int:
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        print("usage: job_wrapper.py <job_key> -- <cmd...>", file=sys.stderr)
        return 2
    job_key = sys.argv[1]
    cmd = sys.argv[3:]
    job_dir = JOBS / job_key
    job_dir.mkdir(parents=True, exist_ok=True)
    if (job_dir / "DONE").exists():
        print(f"job {job_key} already DONE; refusing to rerun", file=sys.stderr)
        return 3
    state_path = job_dir / "state.json"
    if state_path.exists():
        old = json.loads(state_path.read_text(encoding="utf-8"))
        for key in ("wrapper_pid", "child_pid"):
            pid = old.get(key)
            if pid and psutil.pid_exists(pid) and old.get("status") == "running":
                print(f"job {job_key} already running (pid {pid}); refusing second writer", file=sys.stderr)
                return 4
    live = other_live_process(job_key)
    if live:
        print(f"job {job_key} has live wrapper processes {live}; refusing second writer", file=sys.stderr)
        return 4

    state = {
        "job_key": job_key,
        "cmd": cmd,
        "cwd": str(OUT / "code"),
        "status": "starting",
        "wrapper_pid": os.getpid(),
        "started_unix": time.time(),
        "started_local": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_state(job_dir, state)
    with (job_dir / "stdout.log").open("w", encoding="utf-8") as out, (job_dir / "stderr.log").open("w", encoding="utf-8") as err:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        child = subprocess.Popen(cmd, cwd=str(OUT / "code"), stdout=out, stderr=err, env=env)
        state.update({"status": "running", "child_pid": child.pid})
        (job_dir / "pid").write_text(f"wrapper {os.getpid()}\nchild {child.pid}\n", encoding="utf-8")
        write_state(job_dir, state)
        rc = child.wait()
    state.update({
        "status": "succeeded" if rc == 0 else "failed",
        "exit_code": rc,
        "finished_unix": time.time(),
        "finished_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_seconds": time.time() - state["started_unix"],
    })
    (job_dir / "exit_code").write_text(f"{rc}\n", encoding="utf-8")
    write_state(job_dir, state)
    (job_dir / "DONE").write_text(json.dumps({"exit_code": rc, "finished_local": state["finished_local"]}) + "\n", encoding="utf-8")
    return rc


if __name__ == "__main__":
    sys.exit(main())
