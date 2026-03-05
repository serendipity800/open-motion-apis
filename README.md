# 🎭 open-motion-apis

<div align="center">

**A unified OpenAI-style serving infrastructure for text-to-motion generation models.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Models](https://img.shields.io/badge/Models-4%20SOTA-orange)](docs/models.md)

*Generate human motion from text. Score it. Train with it. All through one clean API.*

</div>

---

## 🌟 What is this?

`open-motion-apis` wraps four state-of-the-art text-to-motion models — **MoMask**, **MDM**, **MLD**, and **T2M-GPT** — behind a single, consistent HTTP interface inspired by the OpenAI API design. Instead of wrestling with each model's bespoke inference scripts, data formats, and normalization conventions, you get one schema and one client.

The core insight is a **two-stage architecture**:

```
POST /v1/motion/generate  →  { id: "uuid4", num_frames: 120, model: "t2m_gpt" }
POST /v1/motion/reward    →  { reward: 0.9955, mm_dist: 1.02 }
```

Generation and scoring are **decoupled** by design. A motion is generated once and stored by UUID. The same motion can then be scored against any number of text prompts at any time — which is exactly what you need when running parallel RL rollouts where agents generate motions now and compute rewards later.

---

## 🚀 Key Features

- 🎯 **Unified schema** — one request format, one response format, four models
- 🔑 **UUID4 motion store** — collision-safe intermediate storage for parallel RL environments
- 🤸 **Normalized outputs** — all models produce `(T, 22, 3)` world-space joint positions (HumanML3D convention)
- ⚡ **Decoupled generate → reward** — score the same motion against multiple prompts without regenerating
- 🏆 **SOTA coverage** — MoMask, MDM, MLD, T2M-GPT all supported and benchmarked
- 🐍 **Python client** — `MotionClient` wraps everything in a clean Python API
- 🧪 **Integration tests** — full test suite covering generate, reward, and error paths
- 🐳 **Docker support** — compose file for deploying the store server + embedding service

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     🏪 motion_store_server                      │
│                        :8090  (CPU)                             │
│                                                                 │
│   POST /v1/motion/generate                                      │
│        │                                                        │
│        ├──▶ 🎭 MoMask   :8081 (cuda:0) ──▶ (T, 22, 3)         │
│        ├──▶ 🌊 MDM      :8082 (cuda:1) ──▶ (T, 22, 3)         │
│        ├──▶ 🌀 MLD      :8083 (cuda:N) ──▶ (T, 22, 3)         │
│        └──▶ 🤖 T2M-GPT  :8084 (cuda:0) ──▶ (T, 22, 3)         │
│                                                                 │
│        💾 saves joints as /tmp/motion_store/{uuid}.npy         │
│        📋 saves meta   as /tmp/motion_store/{uuid}.json        │
│        🔑 returns UUID to caller                               │
│                                                                 │
│   POST /v1/motion/reward                                        │
│        │                                                        │
│        └──▶ 🧠 embedding service :9535                         │
│             📐 cosine_sim(text_emb, motion_emb)                │
└─────────────────────────────────────────────────────────────────┘
          ▲                              ▲
          │  1️⃣  generate(prompt)        │  2️⃣  reward(uuid, prompt)
          └──────────── 🤖 RL Agent ─────┘
```

### 🔑 Why UUID4?

With `N` parallel RL agents all generating motions simultaneously, a simple counter (`1, 2, 3...`) causes race conditions and overwrites. UUID4 is a 128-bit random identifier — the collision probability across one billion simultaneous generations is approximately **10⁻¹⁹**. No locks, no coordination, no shared state.

---

## 📊 Supported Models & Benchmark Results

Results on the **HumanML3D** test set. All numbers reproduced from original papers. Our serving layer introduces zero accuracy delta.

| Model | FID ↓ | R@1 ↑ | R@2 ↑ | R@3 ↑ | MM-Dist ↓ | Diversity ↑ | MModality ↑ |
|---|---|---|---|---|---|---|---|
| 📊 Real data | 0.002 | 0.511 | 0.703 | 0.797 | 2.974 | 9.503 | — |
| 🎭 [MoMask](https://ericguo5513.github.io/momask) | **0.228** | **0.521** | **0.713** | **0.807** | **2.958** | 9.609 | **2.713** |
| 🤖 [T2M-GPT](https://mael-zys.github.io/T2M-GPT) | 0.116 | 0.492 | 0.683 | 0.775 | 3.118 | **9.761** | 1.856 |
| 🌀 [MLD](https://chenxin.tech/mld) | 0.473 | 0.481 | 0.673 | 0.772 | 3.196 | 9.724 | 2.413 |
| 🌊 [MDM](https://guytevet.github.io/mdm-page) | 0.544 | 0.320 | 0.498 | 0.611 | 5.566 | 9.559 | 2.799 |

> 💡 **Tip**: For RL training, we recommend **T2M-GPT** as the default backend — it offers the best balance of generation speed, quality, and diversity.

---

## 🏗️ Repository Layout

```
open-motion-apis/
│
├── 📦 motion_api/              # Backend serving layer
│   ├── schemas.py              # Pydantic request/response types (OpenAI-style)
│   ├── config.py               # Port assignments, default sampling parameters
│   ├── utils.py                # encode_motion / decode_motion / save_npz
│   ├── server_base.py          # FastAPI app factory (shared by all backends)
│   ├── client.py               # MotionClient — Python client library
│   └── backends/               # Per-model server implementations
│       ├── momask_server.py    # MoMask backend (port 8081)
│       ├── mdm_server.py       # MDM backend    (port 8082)
│       ├── mld_server.py       # MLD backend    (port 8083)
│       └── t2m_gpt_server.py   # T2M-GPT backend (port 8084)
│
├── 🏪 motion_store/            # UUID-keyed storage + reward computation
│   └── server.py               # Two-endpoint FastAPI server (port 8090)
│
├── 📐 eval/                    # Offline evaluation utilities
│   └── metrics.py              # FID, R Precision, MM-Dist, Diversity, MModality
│
├── 📖 examples/                # Runnable examples
│   ├── basic_generate.py       # Generate a single motion and save to NPZ
│   └── rl_reward_loop.py       # Generate → store → reward RL loop demo
│
├── 🧪 tests/                   # Integration test suite
│   └── test_api.py             # Health, generate, reward, error path tests
│
├── 🐳 docker/                  # Container deployment
│   ├── Dockerfile.api          # Store server image
│   └── docker-compose.yml      # Full stack: store + embedding service
│
├── 🔧 scripts/
│   └── start_all.sh            # Launch all 5 servers in one command
│
├── pyproject.toml              # Package metadata + dependencies
└── LICENSE                     # MIT
```

---

## ⚡ Quickstart

### 1️⃣ Start the backends

Each model runs in its own conda environment. Launch them from their respective directories:

```bash
# 🎭 MoMask
cd /path/to/momask-codes
conda run -n momask python ../open-motion-apis/motion_api/backends/momask_server.py \
    --port 8081 --device cuda:0 --dataset t2m

# 🌊 MDM
cd /path/to/motion-diffusion-model
conda run -n mdm python ../open-motion-apis/motion_api/backends/mdm_server.py \
    --port 8082 --device cuda:1 \
    --model-path ./save/humanml_enc_512_50steps/model000750000.pt

# 🌀 MLD
cd /path/to/motion-latent-diffusion
conda run -n mld python ../open-motion-apis/motion_api/backends/mld_server.py \
    --port 8083 --device cuda:2 \
    --cfg ./configs/config_mld_humanml3d.yaml

# 🤖 T2M-GPT
cd /path/to/T2M-GPT
conda run -n T2M-GPT python ../open-motion-apis/motion_api/backends/t2m_gpt_server.py \
    --port 8084 --device cuda:0
```

Or start everything at once:

```bash
bash scripts/start_all.sh --device cuda:0
```

### 2️⃣ Start the store server

```bash
python -m motion_store.server --port 8090
```

### 3️⃣ Generate & score

```bash
# Generate
curl -X POST http://localhost:8090/v1/motion/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "a person waves their right hand", "model": "t2m_gpt"}'
# → {"id": "3f7a...", "num_frames": 120, "model": "t2m_gpt", "prompt": "..."}

# Score
curl -X POST http://localhost:8090/v1/motion/reward \
  -H 'Content-Type: application/json' \
  -d '{"id": "3f7a...", "prompt": "a person waves their right hand"}'
# → {"id": "3f7a...", "reward": 0.9921, "mm_dist": 1.08}
```

---

## 🐍 Python Client

```python
from motion_api.client import MotionClient

# 🔌 Connect — defaults to localhost:{8081..8084}
# Override for remote servers:
client = MotionClient(servers={
    "t2m_gpt": "http://192.168.1.10:8084",
    "momask":  "http://192.168.1.10:8081",
})

# 🎬 Generate
result = client.generate(
    model="t2m_gpt",
    prompt="a person waves their right hand and bows",
    motion_length=4.0,
    seed=0,
)

print(result.motions[0].shape)   # (80, 22, 3)
print(result.lengths)            # [80]

# 💾 Save in unified NPZ format
result.save_npz("output.npz")

# 🩺 Health check
print(client.health("t2m_gpt"))  # {"status": "ok", "model": "t2m_gpt"}
```

---

## 🤖 RL Integration

The two-endpoint design is built for reinforcement learning. The agent generates a motion (gets a UUID back instantly), does other work, then calls reward when needed:

```python
import requests

BASE = "http://192.168.1.10:8090"

# Inside your rollout loop ────────────────────────────────────────
# 🎬 Step 1: generate motion, get UUID
resp = requests.post(f"{BASE}/v1/motion/generate", json={
    "prompt": agent_output_text,
    "model": "t2m_gpt",
    "seed": episode_seed,
})
motion_id = resp.json()["id"]

# ... agent continues doing other things ...

# 🏆 Step 2: compute reward (can use a different prompt for reward shaping)
resp = requests.post(f"{BASE}/v1/motion/reward", json={
    "id": motion_id,
    "prompt": target_description,   # ← can differ from generation prompt
})
reward  = resp.json()["reward"]     # cosine similarity ∈ [-1, 1]
mm_dist = resp.json()["mm_dist"]    # L2 distance (lower = better)
# ─────────────────────────────────────────────────────────────────
```

### 🔄 Reward Signal Explained

| Metric | Formula | Range | Use as reward? |
|---|---|---|---|
| `reward` | cos(text_emb, motion_emb) | [-1, 1] | ✅ Direct reward |
| `mm_dist` | ‖text_emb − motion_emb‖₂ | [0, ∞) | ✅ Negated penalty |

Both metrics come from a shared motion-language embedding space. They are differentiable proxies for the **MM-Dist** metric used in HumanML3D evaluation, making them principled reward signals for text-conditioned motion generation fine-tuning.

---

## 📡 API Reference

### `POST /v1/motion/generate`

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | str | required | Natural language motion description |
| `model` | str | `"momask"` | `momask` / `mdm` / `mld` / `t2m_gpt` |
| `motion_length` | float | `6.0` | Duration in seconds |
| `num_samples` | int | `1` | Number of samples to generate |
| `seed` | int | `42` | Random seed for reproducibility |
| `sampling_params` | object | `{}` | Model-specific overrides (see below) |

**`sampling_params` reference:**

| Param | Models | Default | Description |
|---|---|---|---|
| `cond_scale` | momask, mdm | `4.0` | Classifier-free guidance scale |
| `temperature` | momask, t2m_gpt | `1.0` | Sampling temperature |
| `topkr` | momask | `0.9` | Top-k filter threshold |
| `time_steps` | momask | `18` | Mask transformer timesteps |
| `guidance_param` | mdm | `2.5` | MDM guidance parameter |
| `guidance_scale` | mld | `7.5` | MLD guidance scale |
| `if_categorial` | t2m_gpt | `true` | Use categorical sampling |

### `POST /v1/motion/reward`

| Field | Type | Description |
|---|---|---|
| `id` | str | UUID from `/v1/motion/generate` |
| `prompt` | str | Text to score against (can differ from generation prompt) |

### `GET /health`

Returns `{"status": "ok", "models": [...], "stored_motions": N}`.

---

## 🗂️ Data Format

All motion data is normalized to the **HumanML3D** convention:

- **Shape**: `(T, 22, 3)` — T frames, 22 SMPL joints, XYZ world coordinates
- **FPS**: 20
- **Encoding**: base64-encoded `.npy` (float32)

**NPZ output format** (from `client.save_npz()`):

```python
import numpy as np
data = np.load("output.npz", allow_pickle=True)
data["texts"]    # (N,)       array of prompt strings
data["motions"]  # (N,)       array of (T, 22, 3) arrays
data["lengths"]  # (N,) int32 array of frame counts
```

---

## 🧪 Running Tests

```bash
# Start the store server first
python -m motion_store.server --port 8090

# Run tests
pytest tests/ -v
```

---

## 🐳 Docker Deployment

```bash
cd docker
docker compose up --build
```

This starts:
- 🏪 `motion-store` on port `8090`
- 🧠 `embedding` service on port `9535`

---

## 📦 Component READMEs

| Component | README |
|---|---|
| 🎭 Backend servers | [motion_api/README.md](motion_api/README.md) |
| 🏪 Motion store | [motion_store/README.md](motion_store/README.md) |
| 📐 Evaluation | [eval/README.md](eval/README.md) |
| 📖 Examples | [examples/README.md](examples/README.md) |

---

## 📜 Citation

If you use this infrastructure in your research, please also cite the underlying model papers:

```bibtex
@article{guo2023momask,
  title={MoMask: Generative Masked Modeling of 3D Human Motions},
  author={Guo, Chuan and Mu, Yuxuan and Javed, Muhammad Gohar and Wang, Sen and Cheng, Li},
  journal={arXiv preprint arXiv:2312.00063}, year={2023}
}
@inproceedings{tevet2022human,
  title={Human Motion Diffusion Model},
  author={Tevet, Guy and Raab, Sigal and Gordon, Brian and Shafir, Yonatan and Cohen-Or, Daniel and Bermano, Amit H},
  booktitle={ICLR}, year={2023}
}
@inproceedings{zhou2023t2m,
  title={T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations},
  author={Zhou, Jianrong and Gu, Jialong and Wang, Xinyu and Liu, Yiwei and Lu, Tao and Ma, Lizhen and others},
  booktitle={CVPR}, year={2023}
}
@inproceedings{chen2023mld,
  title={Executing your Commands via Motion Diffusion in Latent Space},
  author={Chen, Xin and Jiang, Biao and Liu, Wen and Huang, Zilong and Fu, Bin and Chen, Tao and Yu, Jingyi and Yu, Gang},
  booktitle={CVPR}, year={2023}
}
```

---

## 📄 License

MIT — see [LICENSE](LICENSE).
