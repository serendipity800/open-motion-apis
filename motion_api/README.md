# 📦 motion_api — Backend Serving Layer

This package contains the **per-model FastAPI servers** that wrap each text-to-motion model behind a unified HTTP interface, plus the shared utilities (schemas, client, app factory) that tie them together.

---

## 🗂️ Structure

```
motion_api/
├── schemas.py        # 📋 Pydantic request/response types
├── config.py         # ⚙️  Port assignments + default sampling params
├── utils.py          # 🔧 encode/decode/save utilities
├── server_base.py    # 🏭 FastAPI app factory
├── client.py         # 🐍 Python client library
└── backends/
    ├── momask_server.py     # 🎭 MoMask   — port 8081
    ├── mdm_server.py        # 🌊 MDM      — port 8082
    ├── mld_server.py        # 🌀 MLD      — port 8083
    └── t2m_gpt_server.py    # 🤖 T2M-GPT — port 8084
```

---

## 📋 Schema

All backends share the same request/response schema defined in `schemas.py`.

### Request: `GenerateRequest`

```python
class GenerateRequest(BaseModel):
    prompt: str                          # "a person walks forward slowly"
    num_samples: int = 1
    seed: int = 42
    motion_length: float = 6.0          # seconds
    sampling_params: SamplingParams = SamplingParams()
```

### Response: `GenerateResponse`

```python
class GenerateResponse(BaseModel):
    id: str                             # "gen_abc123def456"
    model: str                          # "t2m_gpt"
    prompt: str
    created: int                        # Unix timestamp
    choices: List[MotionChoice]
    usage: Dict[str, Any]              # {"generation_frames": 120}
```

### Motion payload: `MotionData`

```python
class MotionData(BaseModel):
    num_frames: int                     # 120
    num_joints: int                     # 22
    fps: int                            # 20
    data: str                           # base64-encoded (T, 22, 3) float32 npy
```

---

## ⚙️ Sampling Parameters

Each model exposes a subset of the unified `SamplingParams`:

| Parameter | 🎭 MoMask | 🌊 MDM | 🌀 MLD | 🤖 T2M-GPT | Default |
|---|---|---|---|---|---|
| `cond_scale` | ✅ | ✅ | — | — | 4.0 |
| `temperature` | ✅ | — | — | ✅ | 1.0 |
| `topkr` | ✅ | — | — | — | 0.9 |
| `time_steps` | ✅ | — | — | — | 18 |
| `gumbel_sample` | ✅ | — | — | — | false |
| `guidance_param` | — | ✅ | — | — | 2.5 |
| `guidance_scale` | — | — | ✅ | — | 7.5 |
| `sample_mean` | — | — | ✅ | — | false |
| `if_categorial` | — | — | — | ✅ | true |

---

## 🔄 Output Normalization

Each model produces motion in a different native format. All backends normalize to `(T, 22, 3)` world-space joint positions (HumanML3D convention):

| Model | Native Format | Conversion Path |
|---|---|---|
| 🎭 MoMask | `(B, T, 263)` RIC features | `inv_transform` → `recover_from_ric` |
| 🌊 MDM | `(B, J, 6, T)` rot6d / hml_vec | `inv_transform` → `recover_from_ric` → `rot2xyz` |
| 🌀 MLD | list of `(T, 22, 3)` tensors | direct (already xyz) |
| 🤖 T2M-GPT | `(T, 263)` RIC features | `inv_transform` → `recover_from_ric` |

`recover_from_ric` is imported from each model's own codebase (not re-implemented here) to ensure correctness of quaternion operations.

---

## 🏭 App Factory

`server_base.py` provides `create_app(model_name, generate_fn)` which registers:

- `GET /health` → `{"status": "ok", "model": "..."}`
- `GET /v1/models` → `{"id": "...", "default_params": {...}}`
- `POST /v1/motion/generate` → calls `generate_fn(req)` in a thread pool

Each backend's `__main__` block is exactly:

```python
load_model(args)
app = create_app("model_name", generate)
uvicorn.run(app, host=args.host, port=args.port)
```

---

## 🐍 Python Client

`MotionClient` wraps all backends:

```python
from motion_api.client import MotionClient

client = MotionClient()  # localhost by default

result = client.generate("momask", "a person jumps", motion_length=3.0)
print(result.motions[0].shape)   # (60, 22, 3)
result.save_npz("jump.npz")
```

---

## 🚀 Running a Backend

Each server must be launched from the **model's own directory** (so that model-local imports resolve correctly):

```bash
# 🎭 MoMask
cd /path/to/momask-codes
python ../open-motion-apis/motion_api/backends/momask_server.py \
    --port 8081 --device cuda:0 --dataset t2m

# 🤖 T2M-GPT
cd /path/to/T2M-GPT
python ../open-motion-apis/motion_api/backends/t2m_gpt_server.py \
    --port 8084 --device cuda:0
```
