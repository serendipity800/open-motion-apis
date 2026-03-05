# 📖 Examples

Runnable examples demonstrating common usage patterns.

---

## 🎬 basic_generate.py

Generate a single motion from text and save to NPZ.

```bash
python examples/basic_generate.py
```

Shows:
- Connecting to a backend server
- Generating with `MotionClient`
- Inspecting output shapes
- Saving to unified NPZ format

---

## 🤖 rl_reward_loop.py

Minimal demonstration of the **generate → store → reward** loop used in RL training.

```bash
python examples/rl_reward_loop.py \
    --host 192.168.1.10 \
    --port 8090 \
    --prompt "a person waves their right hand" \
    --model t2m_gpt
```

Shows:
- Posting to `/v1/motion/generate` and capturing the UUID
- Posting to `/v1/motion/reward` with the same and different prompts
- Interpreting `reward` (cosine similarity) and `mm_dist` (L2 distance)

**Expected output:**

```
Generated: a4c1f832-7e3b-4a12-b901-0d5e6f7a8b9c  (120 frames)
Reward  : 0.9955  (cosine similarity)
MM-Dist : 1.0249  (L2 distance)
```
