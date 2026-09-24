"""Running jobs in parallel and saving results."""

import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm.auto import tqdm

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def parallel_map(fn, jobs, workers=1, desc=None):
    """Return [fn(job) for job in jobs], optionally spread over processes.

    Every job carries its own seed, so the results do not depend on ``workers``.
    """
    jobs = list(jobs)
    if workers <= 1:
        return [fn(job) for job in tqdm(jobs, desc=desc)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(tqdm(pool.map(fn, jobs), total=len(jobs), desc=desc))


def workers_from_argv():
    """Read the number of worker processes from ``--workers`` (default 1)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args().workers


def save_results(name, results):
    """Write results to results/<name>.pkl."""
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"Wrote results/{name}.pkl")


def load_results(name):
    """Read results/<name>.pkl."""
    with open(RESULTS_DIR / f"{name}.pkl", "rb") as f:
        return pickle.load(f)
