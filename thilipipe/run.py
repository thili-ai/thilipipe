"""Start and stop the five services from a notebook cell.

Processes, not containers. The course workbench is already a container and docker-in-docker would
make this drill undeployable — but the thing that matters pedagogically is the *process and socket
boundary*, and localhost gives you that. What it does not give you is the network, which is why
lesson 3 has to say out loud that these latencies are a floor, not an estimate.
"""

from __future__ import annotations

import subprocess
import sys
import time

import httpx

SERVICES = [
    ("candidates", "thilipipe.services.candidates:app", 8801),
    ("features", "thilipipe.services.features:app", 8802),
    ("filter", "thilipipe.services.filter:app", 8803),
    ("rank", "thilipipe.services.rank:app", 8804),
    ("orchestrator", "thilipipe.orchestrator:app", 8800),
]

_procs: list[subprocess.Popen] = []


def start(wait: float = 40.0) -> None:
    stop()
    for name, target, port in SERVICES:
        _procs.append(subprocess.Popen(
            [sys.executable, "-m", "uvicorn", target, "--host", "127.0.0.1",
             "--port", str(port), "--log-level", "warning"],
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        ))
    deadline = time.time() + wait
    for name, _, port in SERVICES:
        while time.time() < deadline:
            try:
                httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
                break
            except Exception:                                     # noqa: BLE001
                time.sleep(0.3)
        else:
            stop()
            raise RuntimeError(f"{name} did not come up on :{port} — run bootstrap first?")
    print(f"up: {', '.join(n for n, _, _ in SERVICES)}")


def stop() -> None:
    for p in _procs:
        p.terminate()
    for p in _procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    _procs.clear()


if __name__ == "__main__":
    start()
    print("ctrl-c to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop()
