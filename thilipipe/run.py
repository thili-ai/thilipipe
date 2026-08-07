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


def is_up() -> bool:
    """True when the orchestrator answers. Cheap enough to call at the top of every lesson."""
    try:
        return httpx.get("http://127.0.0.1:8800/health", timeout=2.0).status_code == 200
    except Exception:                                             # noqa: BLE001
        return False


def ensure(wait: float = 40.0) -> None:
    """Start the services only if they are not already running.

    Lessons are opened out of order and re-run repeatedly, so `start()` at the top of every
    notebook would kill a system the previous cell was using. This is the idempotent version.
    """
    if is_up():
        print("already up")
        return
    start(wait=wait)


def ensure_bootstrapped() -> None:
    """Train the artifacts if they are missing. Roughly two minutes on CPU, once."""
    from .state import state_dir

    needed = ["item_vectors.npy", "item_vectors_stale.npy", "ranker.pt",
              "train_mask.npy", "test_mask.npy", "offline_recs.json", "meta.json"]
    missing = [f for f in needed if not (state_dir() / f).exists()]
    if not missing:
        print("artifacts present")
        return
    print(f"missing {len(missing)} artifact(s) — training (~2 min)")
    subprocess.run([sys.executable, "-m", "thilipipe.bootstrap"], check=True)


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
