# 🏪 motion_store — UUID-Keyed Motion Storage & Reward Server

The `motion_store` server is the **public-facing entry point** of the whole system. It is the only component that external clients (e.g., RL agents on a remote machine) need to talk to.

It exposes two endpoints that together implement a **generate → store → reward** workflow designed for parallel reinforcement learning training.

---

## 🎯 Design Goals

| Goal | How |
|---|---|
| 🔑 **Collision-free across N parallel agents** | UUID4 identifiers (128-bit random, collision prob ≈ 10⁻¹⁹) |
| ⏱️ **Decoupled generation and scoring** | Generate now, score later — IDs are stable until TTL |
| 🔁 **Score same motion with multiple prompts** | Reward endpoint accepts any text, motion is loaded from disk |
| 💾 **No memory bloat** | Background TTL thread deletes files older than 60 minutes |
| 🚫 **No shared state between agents** | Each agent writes its own UUID files, zero coordination |

---

## 📡 Endpoints

### `POST /v1/motion/generate`

Calls a backend model, stores the result, returns a UUID.

```
Request  →  { prompt, model, motion_length, num_samples, seed }
                ↓
         calls localhost:808x/v1/motion/generate
                ↓
         decodes joints (T, 22, 3) from base64 response
                ↓
         writes /tmp/motion_store/{uuid}.npy   ← joint positions
         writes /tmp/motion_store/{uuid}.json  ← metadata
                ↓
Response →  { id, model, num_frames, prompt }
```

**Supported models:**

| Value | Backend port | Notes |
|---|---|---|
| `"momask"` | 8081 | Best FID |
| `"mdm"` | 8082 | Highest MModality |
| `"mld"` | 8083 | Latent diffusion |
| `"t2m_gpt"` | 8084 | ⭐ Recommended — fastest, most stable |

---

### `POST /v1/motion/reward`

Loads a stored motion by UUID, calls the embedding service, returns reward.

```
Request  →  { id, prompt }
                ↓
         loads /tmp/motion_store/{uuid}.npy
                ↓
         POST http://embedding:9535/extract_embeddings
              { texts: [prompt], joints: [joints.tolist()] }
                ↓
         text_emb   ← 512-dim CLIP-based text embedding
         motion_emb ← 512-dim motion embedding
                ↓
         reward  = cosine_sim(text_emb / ‖t‖, motion_emb / ‖m‖)
         mm_dist = ‖text_emb − motion_emb‖₂
                ↓
Response →  { id, reward, mm_dist }
```

**Reward interpretation:**

| Metric | Range | Interpretation |
|---|---|---|
| `reward` | [-1, 1] | Higher → motion better matches text |
| `mm_dist` | [0, ∞) | Lower → motion better matches text |

---

### `GET /health`

```json
{
  "status": "ok",
  "models": ["momask", "mdm", "mld", "t2m_gpt"],
  "stored_motions": 42
}
```

---

## 💾 Storage Layout

```
/tmp/motion_store/             ← configurable via --store-dir
├── 3f7a1b2c-9e4d-....npy     ← np.ndarray, (T, 22, 3), float32
├── 3f7a1b2c-9e4d-....json    ← {"prompt": "...", "model": "t2m_gpt",
│                                 "num_frames": 120, "created_at": 1234567890.0}
├── a5f1dc86-b9e9-....npy
├── a5f1dc86-b9e9-....json
└── ...
```

---

## 🕐 TTL Cleanup

A background daemon thread wakes every 60 seconds and removes any file whose `mtime` is older than `--ttl-minutes` (default: 60). Both the `.npy` and `.json` files are removed together.

This ensures the store doesn't accumulate gigabytes of stale motion data during long training runs.

---

## 🚀 Running

```bash
# Minimal
python -m motion_store.server --port 8090

# Custom store directory + TTL
python -m motion_store.server \
    --port 8090 \
    --store-dir /fast_nvme/motion_store \
    --ttl-minutes 120

# Check it's running
curl http://localhost:8090/health
```

---

## 📝 Example: Full Workflow

```python
import requests

BASE = "http://192.168.1.10:8090"

# Agent generates a motion ──────────────────────────────────────
r = requests.post(f"{BASE}/v1/motion/generate", json={
    "prompt": "a person slowly raises both arms overhead",
    "model": "t2m_gpt",
    "motion_length": 4.0,
    "seed": 7,
})
motion_id = r.json()["id"]           # "a4c1f..."
num_frames = r.json()["num_frames"]  # 80

# Agent can do other work here...

# Agent scores the motion ───────────────────────────────────────
r = requests.post(f"{BASE}/v1/motion/reward", json={
    "id": motion_id,
    "prompt": "a person slowly raises both arms overhead",
})
reward  = r.json()["reward"]    # 0.9912
mm_dist = r.json()["mm_dist"]  # 1.08

# Score against a different prompt (reward shaping) ─────────────
r = requests.post(f"{BASE}/v1/motion/reward", json={
    "id": motion_id,
    "prompt": "a person runs forward quickly",
})
print(r.json()["reward"])  # 0.38 — correctly much lower
```
