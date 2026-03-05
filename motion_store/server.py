"""
Motion Store Server — 4090 side

Two-endpoint design:
  POST /motion/generate  → generate motion, save to disk, return UUID
  POST /motion/reward    → load motion by UUID, compute reward, return float

UUID4 guarantees global uniqueness across all parallel agent environments.
Files are stored in --store-dir (default /tmp/motion_store).
A background thread deletes files older than --ttl-minutes (default 60).

Usage:
    python motion_store_server.py [--port 8090] [--store-dir /tmp/motion_store] [--ttl-minutes 60]

Example (from A100 or any machine on same LAN):
    # Generate
    curl -X POST http://192.168.31.90:8090/motion/generate \
      -H 'Content-Type: application/json' \
      -d '{"prompt": "a person walks", "model": "momask"}'
    # -> {"id": "3f7a...", "model": "momask", "num_frames": 120, "prompt": "..."}

    # Reward
    curl -X POST http://192.168.31.90:8090/motion/reward \
      -H 'Content-Type: application/json' \
      -d '{"id": "3f7a...", "prompt": "a person walks"}'
    # -> {"id": "3f7a...", "reward": 0.992, "mm_dist": 1.07}
"""

import argparse
import os
import sys
import threading
import time
import uuid
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from utils import decode_motion

# ── Config ────────────────────────────────────────────────────
MODEL_PORTS = {
    "momask":  8081,
    "mdm":     8082,
    "mld":     8083,
    "t2m_gpt": 8084,
}
EMBD_SERVICE_URL = "http://172.17.0.3:9535/extract_embeddings"
MOTION_API_URL   = "http://127.0.0.1:{port}/v1/motion/generate"

app = FastAPI(title="Motion Store Server")

# Will be set in main()
STORE_DIR: str = "/tmp/motion_store"
TTL_SECONDS: int = 3600


# ── Schemas ───────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str
    model: str = "momask"           # momask | mdm | mld | t2m_gpt
    motion_length: float = 6.0
    num_samples: int = 1
    seed: int = 42


class GenerateResponse(BaseModel):
    id: str                         # UUID4, use this in /motion/reward
    model: str
    num_frames: int
    prompt: str


class RewardRequest(BaseModel):
    id: str                         # UUID from /motion/generate
    prompt: str                     # text to score against (can differ from generation prompt)


class RewardResponse(BaseModel):
    id: str
    reward: float                   # cosine similarity, higher = better
    mm_dist: float                  # L2 distance in embedding space, lower = better


# ── Storage helpers ───────────────────────────────────────────
def _motion_path(motion_id: str) -> str:
    return os.path.join(STORE_DIR, f"{motion_id}.npy")

def _meta_path(motion_id: str) -> str:
    return os.path.join(STORE_DIR, f"{motion_id}.json")

def save_motion(motion_id: str, joints: np.ndarray, meta: dict) -> None:
    np.save(_motion_path(motion_id), joints)
    with open(_meta_path(motion_id), "w") as f:
        json.dump(meta, f)

def load_motion(motion_id: str) -> tuple[np.ndarray, dict]:
    npy_path = _motion_path(motion_id)
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Motion ID '{motion_id}' not found (expired or invalid)")
    joints = np.load(npy_path)
    with open(_meta_path(motion_id)) as f:
        meta = json.load(f)
    return joints, meta


# ── TTL cleanup thread ────────────────────────────────────────
def _cleanup_loop():
    while True:
        time.sleep(60)
        now = time.time()
        try:
            for fname in os.listdir(STORE_DIR):
                fpath = os.path.join(STORE_DIR, fname)
                if os.path.isfile(fpath) and now - os.path.getmtime(fpath) > TTL_SECONDS:
                    os.remove(fpath)
        except Exception:
            pass


# ── Core functions ────────────────────────────────────────────
def _generate_motion(prompt: str, model: str, motion_length: float,
                     num_samples: int, seed: int) -> tuple[np.ndarray, int]:
    port = MODEL_PORTS[model]
    resp = requests.post(
        MOTION_API_URL.format(port=port),
        json={"prompt": prompt, "motion_length": motion_length,
              "num_repetitions": num_samples, "seed": seed},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    joints = decode_motion(data["choices"][0]["motion"]["data"])  # (T, 22, 3)
    num_frames = int(data["choices"][0]["motion"]["num_frames"])
    return joints, num_frames


def _compute_reward(prompt: str, joints: np.ndarray) -> tuple[float, float]:
    payload = {"texts": [prompt], "joints": [joints.tolist()]}
    resp = requests.post(EMBD_SERVICE_URL, json=payload, timeout=60)
    resp.raise_for_status()
    d = resp.json()
    text_emb   = np.array(d["text_embs"][0],   dtype=np.float32)
    motion_emb = np.array(d["motion_embs"][0], dtype=np.float32)
    t_norm = text_emb   / (np.linalg.norm(text_emb)   + 1e-8)
    m_norm = motion_emb / (np.linalg.norm(motion_emb) + 1e-8)
    cosine_sim = float(np.dot(t_norm, m_norm))
    mm_dist    = float(np.linalg.norm(text_emb - motion_emb))
    return cosine_sim, mm_dist


# ── Endpoints ─────────────────────────────────────────────────
@app.post("/motion/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if req.model not in MODEL_PORTS:
        raise HTTPException(400, f"Unknown model '{req.model}'. "
                                 f"Choose from: {list(MODEL_PORTS)}")
    try:
        joints, num_frames = _generate_motion(
            req.prompt, req.model, req.motion_length, req.num_samples, req.seed
        )
    except requests.exceptions.ConnectionError:
        raise HTTPException(503, f"Model server '{req.model}' unreachable "
                                 f"on port {MODEL_PORTS[req.model]}")
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {e}")

    motion_id = str(uuid.uuid4())
    save_motion(motion_id, joints, {
        "prompt": req.prompt,
        "model": req.model,
        "num_frames": num_frames,
        "created_at": time.time(),
    })
    return GenerateResponse(
        id=motion_id,
        model=req.model,
        num_frames=num_frames,
        prompt=req.prompt,
    )


@app.post("/motion/reward", response_model=RewardResponse)
def reward(req: RewardRequest):
    try:
        joints, _ = load_motion(req.id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to load motion: {e}")

    try:
        cosine_sim, mm_dist = _compute_reward(req.prompt, joints)
    except Exception as e:
        raise HTTPException(500, f"Reward computation failed: {e}")

    return RewardResponse(id=req.id, reward=cosine_sim, mm_dist=mm_dist)


@app.get("/health")
def health():
    files = len([f for f in os.listdir(STORE_DIR) if f.endswith(".npy")])
    return {"status": "ok", "models": list(MODEL_PORTS), "stored_motions": files}


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--store-dir", default="/tmp/motion_store")
    parser.add_argument("--ttl-minutes", type=int, default=60)
    args = parser.parse_args()

    STORE_DIR = args.store_dir
    TTL_SECONDS = args.ttl_minutes * 60
    os.makedirs(STORE_DIR, exist_ok=True)

    t = threading.Thread(target=_cleanup_loop, daemon=True)
    t.start()

    print(f"Motion store: {STORE_DIR} (TTL={args.ttl_minutes}min)")
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
