"""Run a per-tilt-series function over many tilt series on one or more GPUs.

Each tilt series runs in its own short-lived worker process (a crash cannot take others down) with its GPU
chosen round-robin and its CPU math threads limited (see :func:`jolt.utils.set_thread_limits`).
"""

from __future__ import annotations

import multiprocessing
import os
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass


@dataclass
class Task:
    """One unit of work: ``function(*args)`` for the tilt series ``name``."""

    name: str
    function: Callable
    args: tuple


def visible_gpus(requested: str | None = None) -> list[str]:
    """GPU ids to use: ``requested`` (comma-separated), else all GPUs torch can see, else none (CPU)."""
    if requested:
        if requested.strip().lower() == "cpu":
            return []
        return [g.strip() for g in requested.split(",") if g.strip()]
    try:
        import torch

        return [str(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        return []


def _init_worker(counter, gpus: list[str], worker_threads: int) -> None:
    if gpus:
        with counter.get_lock():
            slot = counter.value
            counter.value += 1
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus[slot % len(gpus)]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from jolt.utils import set_thread_limits

    set_thread_limits(worker_threads)


def _run_task(task: Task) -> dict:
    t0 = time.time()
    try:
        result = task.function(*task.args)
        return {"name": task.name, "ok": True, "result": result, "seconds": round(time.time() - t0, 1)}
    except Exception as exc:
        return {
            "name": task.name,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "seconds": round(time.time() - t0, 1),
        }


def run_tasks(
    tasks: list[Task], gpus: list[str], n_jobs: int | None = None, worker_threads: int = 1, log=print
) -> list[dict]:
    """Run tasks in parallel worker processes and return one outcome per task (in completion order).

    Parameters
    ----------
    tasks : list of Task
    gpus : list of str
        GPU ids to spread the workers over (empty: CPU only).
    n_jobs : int, optional
        Concurrent workers (default: one per GPU, or one on CPU).
    worker_threads : int
        CPU math threads per worker.
    log : callable
        Receives one line per finished task.

    Returns
    -------
    list of dict
        ``name``, ``ok``, ``result`` or ``error``/``traceback``, and ``seconds`` for every task.
    """
    if not tasks:
        return []
    n_jobs = max(1, int(n_jobs or max(1, len(gpus))))
    context = multiprocessing.get_context("spawn")
    counter = context.Value("i", 0)
    outcomes = []
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=min(n_jobs, len(tasks)),
        mp_context=context,
        initializer=_init_worker,
        initargs=(counter, list(gpus), int(worker_threads)),
        max_tasks_per_child=1,
    ) as pool:
        futures = {pool.submit(_run_task, task): task.name for task in tasks}
        for future in as_completed(futures):
            try:
                outcome = future.result()
            except Exception as exc:  # the worker process itself died
                outcome = {"name": futures[future], "ok": False, "error": f"worker died: {exc}", "seconds": None}
            outcomes.append(outcome)
            state = "done" if outcome["ok"] else f"FAILED ({outcome['error']})"
            log(f"[{len(outcomes)}/{len(tasks)}] {outcome['name']}: {state} after {outcome['seconds']} s")
    log(f"{len(tasks)} tilt series in {time.time() - t0:.0f} s")
    return outcomes
