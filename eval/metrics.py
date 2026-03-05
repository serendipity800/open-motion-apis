"""Offline evaluation utilities for text-to-motion models.

Computes standard HumanML3D benchmark metrics:
  - FID (Frechet Inception Distance)
  - R Precision (Top-1 / Top-2 / Top-3)
  - MM-Dist (Motion-Text distance in embedding space)
  - Diversity
  - MModality
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple


def compute_fid(real_feats: np.ndarray, fake_feats: np.ndarray) -> float:
    """Frechet Inception Distance between real and generated motion features."""
    mu_r, sigma_r = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu_f, sigma_f = fake_feats.mean(0), np.cov(fake_feats, rowvar=False)
    diff = mu_r - mu_f
    covmean = _sqrtm(sigma_r @ sigma_f)
    return float(diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean))


def compute_r_precision(
    text_embs: np.ndarray,
    motion_embs: np.ndarray,
    top_k: Tuple[int, ...] = (1, 2, 3),
) -> dict:
    """R Precision: fraction of prompts whose ground-truth motion ranks in top-k."""
    scores = text_embs @ motion_embs.T  # (N, N)
    results = {}
    for k in top_k:
        hits = sum(
            np.argpartition(scores[i], -k)[-k:].tolist().count(i)
            for i in range(len(scores))
        )
        results[f"top_{k}"] = hits / len(scores)
    return results


def compute_mm_dist(text_embs: np.ndarray, motion_embs: np.ndarray) -> float:
    """Mean L2 distance between paired text and motion embeddings."""
    return float(np.linalg.norm(text_embs - motion_embs, axis=-1).mean())


def compute_diversity(motion_feats: np.ndarray, n_pairs: int = 300) -> float:
    """Average pairwise L2 distance across random motion pairs."""
    idx = np.random.choice(len(motion_feats), size=(n_pairs, 2), replace=False)
    return float(np.linalg.norm(motion_feats[idx[:, 0]] - motion_feats[idx[:, 1]], axis=-1).mean())


def _sqrtm(M: np.ndarray) -> np.ndarray:
    from scipy.linalg import sqrtm
    result = sqrtm(M)
    if np.iscomplexobj(result):
        result = result.real
    return result
