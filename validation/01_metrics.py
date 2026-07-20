"""Benchmark clustering metrics on synthetic EEG-like data.

This script simulates random datasets for different true numbers of clusters
and noise levels. For each simulation, it fits ``ModKMeans`` over candidate
values of ``K`` and checks whether each metric selects the true ``K``.

The output is saved as a CSV file with one row per
``(true_k, noise_level, repeat, metric)``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from mne import create_info
from sklearn.datasets import make_blobs

from pycrostates.cluster import ModKMeans
from pycrostates.io import ChData
from pycrostates.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    dunn_score,
    silhouette_score,
)

import pandas as pd


METRICS = {
    "silhouette": (silhouette_score, "max"),
    "calinski_harabasz": (calinski_harabasz_score, "max"),
    "dunn": (dunn_score, "max"),
    "davies_bouldin": (davies_bouldin_score, "min"),
}


def _simulate_data(
    *,
    true_k: int,
    n_channels: int,
    n_samples_per_cluster: int,
    blob_std: float,
    noise_level: float,
    seed: int,
) -> ChData:
    """Create synthetic EEG-like topographies for a single benchmark run."""
    n_samples = true_k * n_samples_per_cluster
    rng = np.random.RandomState(seed)
    x, _ = make_blobs(
        n_samples=n_samples,
        centers=true_k,
        n_features=n_channels,
        cluster_std=blob_std,
        random_state=rng,
    )

    # Add Gaussian noise, random amplitude scaling, and random polarity flips.
    x += rng.normal(scale=noise_level, size=x.shape)
    x *= rng.normal(size=(n_samples, 1)) * 0.5 + 1.0
    signs = np.where(rng.rand(n_samples) < 0.5, -1.0, 1.0)
    x *= signs[:, np.newaxis]

    info = create_info(n_channels, 1000.0, ch_types="eeg")
    return ChData(x.T, info)


def _select_best_k(scores_by_k: dict[int, float], direction: str) -> tuple[int, float]:
    """Return the best K and score according to metric optimization direction."""
    if direction == "max":
        best_k = max(scores_by_k, key=scores_by_k.get)
    else:
        best_k = min(scores_by_k, key=scores_by_k.get)
    return best_k, scores_by_k[best_k]


def run_benchmark(
    *,
    true_k_values: list[int],
    candidate_k_values: list[int],
    noise_levels: list[float],
    repeats: int,
    n_channels: int,
    n_samples_per_cluster: int,
    blob_std: list[float],
    n_jobs: int,
    seed: int,
) -> list[dict[str, float | int | str | bool]]:
    """Run benchmark and return rows for CSV export."""
    rows = []

    for true_k in true_k_values:
        for noise_level in noise_levels:
            for blob_std_value in blob_std:
                for repeat in range(repeats):
                    print(f"Running benchmark for true_k={true_k}, noise_level={noise_level}, standard_deviation={blob_std_value}, repeat={repeat}")
                    run_seed = seed + 10_000 * repeat + 100 * true_k + int(noise_level * 10)
                    data = _simulate_data(
                        true_k=true_k,
                        n_channels=n_channels,
                        n_samples_per_cluster=n_samples_per_cluster,
                        blob_std=blob_std_value,
                        noise_level=noise_level,
                        seed=run_seed,
                    )

                    metric_scores: dict[str, dict[int, float]] = {
                        metric_name: {} for metric_name in METRICS
                    }

                    for candidate_k in candidate_k_values:
                        modk = ModKMeans(n_clusters=candidate_k, random_state=run_seed)
                        modk.fit(data, n_jobs=n_jobs, verbose="WARNING")

                        for metric_name, (metric_fn, _) in METRICS.items():
                            metric_scores[metric_name][candidate_k] = float(metric_fn(modk))

                    for metric_name, (_, direction) in METRICS.items():
                        best_k, best_score = _select_best_k(metric_scores[metric_name], direction)
                        rows.append(
                            {
                                "true_k": true_k,
                                "noise_level": noise_level,
                                "repeat": repeat,
                                "metric": metric_name,
                                "best_k": best_k,
                                "best_score": best_score,
                                "match_true_k": best_k == true_k,
                            }
                        )

    return rows


if __name__ == "__main__":
    true_k_values = [2, 3, 4, 5, 6, 7, 8, 9]
    candidate_k_values = [2, 3, 4, 5, 6, 7, 8, 9]
    noise_levels = [0.1, 0.3, 0.5, 0.7, 1.0]
    repeats = 10
    n_channels = 64
    n_samples_per_cluster = 1200
    blob_std = [1, 5, 10]
    n_jobs = -1

    rows = run_benchmark(
        true_k_values=true_k_values,
        candidate_k_values=candidate_k_values,
        noise_levels=noise_levels,
        repeats=repeats,
        n_channels=n_channels,
        n_samples_per_cluster=n_samples_per_cluster,
        blob_std=blob_std,
        n_jobs=n_jobs,
        seed=42,
    )
    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).parent / "blob_metrics_results.csv", index=False)

