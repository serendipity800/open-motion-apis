# 📐 eval — Offline Evaluation Utilities

This module provides **offline evaluation metrics** for generated motion sequences, following the standard HumanML3D benchmark protocol.

---

## 📊 Metrics

### 🎯 FID (Fréchet Inception Distance)
Measures the distributional distance between real and generated motion features in a pre-trained embedding space. Lower is better.

```python
from eval.metrics import compute_fid
fid = compute_fid(real_feats, generated_feats)  # lower = better
```

### 🎯 R Precision (Top-1 / Top-2 / Top-3)
For each generated motion, its ground-truth text description should rank in the top-k against 31 randomly selected mismatched descriptions. Higher is better.

```python
from eval.metrics import compute_r_precision
scores = compute_r_precision(text_embs, motion_embs, top_k=(1, 2, 3))
# {"top_1": 0.521, "top_2": 0.713, "top_3": 0.807}
```

### 🎯 MM-Dist (Motion-Text Distance)
Mean L2 distance between paired text and motion embeddings in the shared embedding space. Lower is better.

```python
from eval.metrics import compute_mm_dist
dist = compute_mm_dist(text_embs, motion_embs)  # lower = better
```

### 🎯 Diversity
Average pairwise L2 distance between randomly sampled motion feature pairs. Higher means the model generates a wider variety of motions.

```python
from eval.metrics import compute_diversity
div = compute_diversity(motion_feats, n_pairs=300)  # higher = better
```

---

## 📈 Reference Numbers (HumanML3D Test Set)

| Metric | Real | MoMask | T2M-GPT | MLD | MDM |
|---|---|---|---|---|---|
| FID ↓ | 0.002 | **0.228** | 0.116 | 0.473 | 0.544 |
| R@1 ↑ | 0.511 | **0.521** | 0.492 | 0.481 | 0.320 |
| MM-Dist ↓ | 2.974 | **2.958** | 3.118 | 3.196 | 5.566 |
| Diversity ↑ | 9.503 | 9.609 | **9.761** | 9.724 | 9.559 |

---

## 🔧 Usage with MotionClient

```python
import numpy as np
from motion_api.client import MotionClient
from eval.metrics import compute_mm_dist, compute_r_precision

client = MotionClient()
prompts = [...]  # list of text prompts

# Generate motions and collect embeddings
# (embed via your embedding service separately)
text_embs   = np.load("text_embs.npy")    # (N, 512)
motion_embs = np.load("motion_embs.npy")  # (N, 512)

print("MM-Dist:", compute_mm_dist(text_embs, motion_embs))
print("R Prec:", compute_r_precision(text_embs, motion_embs))
```
