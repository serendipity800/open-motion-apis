"""Minimal RL reward loop example using the two-endpoint store server.

Demonstrates the generate → (delay) → reward pattern used in
parallel RL training environments.
"""

import argparse
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--prompt", default="a person walks forward slowly")
    parser.add_argument("--model", default="t2m_gpt")
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"

    # Step 1: generate — returns UUID immediately
    resp = requests.post(f"{base}/v1/motion/generate", json={
        "prompt": args.prompt,
        "model": args.model,
        "motion_length": 6.0,
        "seed": 42,
    })
    resp.raise_for_status()
    data = resp.json()
    motion_id = data["id"]
    print(f"Generated: {motion_id}  ({data['num_frames']} frames)")

    # Step 2: reward — can be called later, even with a different prompt
    resp = requests.post(f"{base}/v1/motion/reward", json={
        "id": motion_id,
        "prompt": args.prompt,
    })
    resp.raise_for_status()
    r = resp.json()
    print(f"Reward  : {r['reward']:.4f}  (cosine similarity)")
    print(f"MM-Dist : {r['mm_dist']:.4f}  (L2 distance)")


if __name__ == "__main__":
    main()
