"""Utility functions: base64 encoding, NPZ I/O.

Each backend imports recover_from_ric from its own model codebase
(which uses the correct quaternion operations), so we don't re-implement it here.
"""

from __future__ import annotations

import base64
import io
from typing import List

import numpy as np


def encode_motion(arr: np.ndarray) -> str:
    """Encode a numpy array (T, 22, 3) to a base64 string (npy format)."""
    arr = arr.astype(np.float32)
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def decode_motion(s: str) -> np.ndarray:
    """Decode a base64 string back to a numpy array."""
    buf = io.BytesIO(base64.b64decode(s))
    return np.load(buf)


def save_npz(path: str, texts: List[str], motions: List[np.ndarray], lengths: List[int]) -> None:
    """Save motions in the unified NPZ format.

    Args:
        path: Output file path (with or without .npz extension).
        texts: List of text prompts, one per motion.
        motions: List of numpy arrays, each of shape (T, 22, 3).
        lengths: List of frame counts.
    """
    np.savez(
        path,
        texts=np.array(texts, dtype=object),
        motions=np.array(motions, dtype=object),
        lengths=np.array(lengths, dtype=np.int32),
    )
