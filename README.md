# open-motion-apis

**A unified OpenAI-style serving layer for text-to-motion models.**

Wrap MoMask, MDM, MLD, and T2M-GPT behind a single consistent HTTP API — then plug them directly into reinforcement learning pipelines, evaluation harnesses, or downstream generation workflows.

```
POST /v1/motion/generate   →  { id, num_frames, model }
POST /v1/motion/reward     →  { reward, mm_dist }
```

---

## Why

Every text-to-motion paper ships its own inference script with its own CLI flags, data formats, and normalization conventions. Connecting them to anything else means writing glue code five times over. `open-motion-apis` does that once:

- **Unified request/response schema** (OpenAI-style, base64-encoded NumPy payloads)
- **Decoupled generate → reward** endpoints with UUID-keyed intermediate storage — designed for parallel RL rollouts where generation and scoring happen at different times
- **No format lock-in**: all outputs normalised to `(T, 22, 3)` world-space joint positions (HumanML3D convention)

---

## Quickstart

```bash
# 1. Start a backend (run from the model's own directory)
cd /path/to/momask-codes
python ../open-motion-apis/motion_api/backends/momask_server.py \
    --port 8081 --device cuda:0 --dataset t2m

# 2. Start the store server
python -m motion_store.server --port 8090

# 3. Generate + score from anywhere on the LAN
python examples/rl_reward_loop.py --host 192.168.1.x --prompt "a person walks forward"
```

Or start everything at once:

```bash
bash scripts/start_all.sh --device cuda:0
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   motion_store_server                │
│                    :8090  (CPU)                      │
│                                                      │
│   POST /v1/motion/generate                           │
│        │                                            │
│        ├──▶ momask  :8081 (cuda:0)                  │
│        ├──▶ mdm     :8082 (cuda:1)                  │
│        ├──▶ mld     :8083 (cuda:N)                  │
│        └──▶ t2m_gpt :8084 (cuda:0)                  │
│                                                      │
│        saves (T,22,3) joints as {uuid}.npy           │
│        returns UUID to caller                        │
│                                                      │
│   POST /v1/motion/reward                             │
│        │                                            │
│        └──▶ embedding service                        │
│             cosine_sim(text_emb, motion_emb)         │
└─────────────────────────────────────────────────────┘
         ▲                        ▲
         │  generate(prompt)      │  reward(uuid, prompt)
         └────────── RL Agent ────┘
```

UUID4 keys mean N parallel agent environments can generate and score independently with zero coordination overhead.

---

## Supported Models

Results on **HumanML3D** test set. All numbers from original papers; our serving layer introduces no accuracy delta.

| Model | FID ↓ | R Precision (Top-1) ↑ | MM-Dist ↓ | Diversity ↑ | Supported |
|---|---|---|---|---|---|
| Real data | 0.002 | 0.511 | 2.974 | 9.503 | — |
| [MoMask](https://ericguo5513.github.io/momask) | **0.228** | **0.521** | **2.958** | 9.609 | ✅ |
| [T2M-GPT](https://mael-zys.github.io/T2M-GPT) | 0.116 | 0.492 | 3.118 | 9.761 | ✅ |
| [MLD](https://chenxin.tech/mld) | 0.473 | 0.481 | 3.196 | 9.724 | ✅ |
| [MDM](https://guytevet.github.io/mdm-page) | 0.544 | 0.320 | 5.566 | 9.559 | ✅ |

---

## Reward Signal

The `/v1/motion/reward` endpoint returns two metrics derived from a shared motion-text embedding space:

| Metric | Formula | Range | Interpretation |
|---|---|---|---|
| `reward` | cosine_sim(text_emb, motion_emb) | [-1, 1] | Higher = more aligned |
| `mm_dist` | ‖text_emb − motion_emb‖₂ | [0, ∞) | Lower = more aligned |

These metrics are differentiable proxies for the MM-Dist evaluation metric used in the HumanML3D benchmark, making them suitable as dense reward signals for RL fine-tuning.

---

## Repo Layout

```
open-motion-apis/
├── motion_api/            # Backend servers (one per model)
│   ├── schemas.py         # Pydantic request/response types
│   ├── config.py          # Port assignments, default sampling params
│   ├── utils.py           # encode_motion / decode_motion / save_npz
│   ├── server_base.py     # FastAPI app factory
│   ├── client.py          # Python client  (MotionClient)
│   └── backends/
│       ├── momask_server.py
│       ├── mdm_server.py
│       ├── mld_server.py
│       └── t2m_gpt_server.py
├── motion_store/          # UUID-keyed intermediate storage + reward
│   └── server.py
├── eval/                  # Offline evaluation utilities
│   └── metrics.py
├── examples/
│   ├── basic_generate.py
│   └── rl_reward_loop.py
├── tests/
│   └── test_api.py
├── docker/
│   ├── Dockerfile.api
│   └── docker-compose.yml
└── scripts/
    └── start_all.sh
```

---

## Python Client

```python
from motion_api.client import MotionClient

client = MotionClient(servers={"t2m_gpt": "http://192.168.1.x:8084"})

result = client.generate(
    model="t2m_gpt",
    prompt="a person waves their right hand",
    motion_length=4.0,
    seed=0,
)

print(result.motions[0].shape)   # (80, 22, 3)
result.save_npz("output.npz")
```

---

## RL Integration

```python
import requests

BASE = "http://192.168.1.x:8090"

# Inside your rollout loop:
motion_id = requests.post(f"{BASE}/v1/motion/generate", json={
    "prompt": agent_output,
    "model": "t2m_gpt",
}).json()["id"]

reward = requests.post(f"{BASE}/v1/motion/reward", json={
    "id": motion_id,
    "prompt": target_description,
}).json()["reward"]
```

---

## Requirements

- Python 3.9+
- PyTorch ≥ 2.0
- `fastapi`, `uvicorn`, `pydantic`, `numpy`, `requests`
- Each model's own conda environment (see individual backend docs)

---

## License

MIT
