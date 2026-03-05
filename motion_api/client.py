"""Python client for the unified Text2Motion API."""

from __future__ import annotations

from typing import Dict, List, Optional, Any

import numpy as np
import requests

from .config import PORTS
from .utils import decode_motion, save_npz


class MotionResponse:
    """Parsed response from a generate call."""

    def __init__(self, model: str, prompt: str, motions: List[np.ndarray], lengths: List[int]):
        self.model = model
        self.prompt = prompt
        self.motions = motions    # list of (T, 22, 3) arrays
        self.lengths = lengths    # list of frame counts

    def save_npz(self, path: str) -> None:
        """Save motions in the unified NPZ format (texts, motions, lengths)."""
        texts = [self.prompt] * len(self.motions)
        save_npz(path, texts, self.motions, self.lengths)

    def __repr__(self) -> str:
        shapes = [m.shape for m in self.motions]
        return (
            f"MotionResponse(model={self.model!r}, prompt={self.prompt!r}, "
            f"motions={shapes}, lengths={self.lengths})"
        )


class MotionClient:
    """Client for the unified Text2Motion API."""

    def __init__(self, servers: Optional[Dict[str, str]] = None):
        if servers is None:
            servers = {name: f"http://localhost:{port}" for name, port in PORTS.items()}
        self.servers = servers

    def _url(self, model: str, path: str) -> str:
        base = self.servers.get(model)
        if base is None:
            raise ValueError(f"Unknown model {model!r}. Available: {list(self.servers)}")
        return base.rstrip("/") + path

    def generate(
        self,
        model: str,
        prompt: str,
        num_samples: int = 1,
        seed: int = 42,
        motion_length: float = 6.0,
        **sampling_params: Any,
    ) -> MotionResponse:
        """Generate motions from a text prompt.

        Args:
            model: Model name, e.g. "momask", "mdm", "mld", "t2m_gpt".
            prompt: Text description of the motion.
            num_samples: Number of samples to generate.
            seed: Random seed.
            motion_length: Desired motion duration in seconds.
            **sampling_params: Optional model-specific sampling parameters
                (cond_scale, temperature, topkr, time_steps, gumbel_sample,
                guidance_param, num_repetitions, guidance_scale, sample_mean,
                if_categorial).

        Returns:
            A :class:`MotionResponse` with the generated motions.
        """
        payload = {
            "prompt": prompt,
            "num_samples": num_samples,
            "seed": seed,
            "motion_length": motion_length,
            "sampling_params": sampling_params,
        }
        resp = requests.post(self._url(model, "/v1/motion/generate"), json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()

        motions = []
        lengths = []
        for choice in data["choices"]:
            arr = decode_motion(choice["motion"]["data"])  # (T, 22, 3)
            motions.append(arr)
            lengths.append(choice["motion"]["num_frames"])

        return MotionResponse(
            model=data["model"],
            prompt=data["prompt"],
            motions=motions,
            lengths=lengths,
        )

    def health(self, model: str) -> dict:
        """Check server health.

        Returns:
            dict with keys ``status`` and ``model``.
        """
        resp = requests.get(self._url(model, "/health"), timeout=10)
        resp.raise_for_status()
        return resp.json()

    def model_info(self, model: str) -> dict:
        """Retrieve model info and default parameters."""
        resp = requests.get(self._url(model, "/v1/models"), timeout=10)
        resp.raise_for_status()
        return resp.json()
