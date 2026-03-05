"""Integration tests for motion_store_server endpoints."""

import pytest
import requests

BASE = "http://127.0.0.1:8090"


def test_health():
    r = requests.get(f"{BASE}/health", timeout=5)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_generate_and_reward():
    prompt = "a person walks forward slowly"
    r = requests.post(f"{BASE}/v1/motion/generate", json={
        "prompt": prompt, "model": "t2m_gpt", "seed": 0,
    }, timeout=120)
    assert r.status_code == 200
    motion_id = r.json()["id"]
    assert len(motion_id) == 36  # UUID4

    r = requests.post(f"{BASE}/v1/motion/reward", json={
        "id": motion_id, "prompt": prompt,
    }, timeout=30)
    assert r.status_code == 200
    assert -1.0 <= r.json()["reward"] <= 1.0


def test_reward_sanity():
    """Same-prompt reward should exceed cross-prompt reward."""
    prompt_a = "a person walks forward slowly"
    prompt_b = "a person does a cartwheel"
    motion_id = requests.post(f"{BASE}/v1/motion/generate", json={
        "prompt": prompt_a, "model": "t2m_gpt", "seed": 1,
    }, timeout=120).json()["id"]

    reward_a = requests.post(f"{BASE}/v1/motion/reward", json={"id": motion_id, "prompt": prompt_a}).json()["reward"]
    reward_b = requests.post(f"{BASE}/v1/motion/reward", json={"id": motion_id, "prompt": prompt_b}).json()["reward"]
    assert reward_a > reward_b


def test_invalid_id_returns_404():
    r = requests.post(f"{BASE}/v1/motion/reward", json={
        "id": "00000000-dead-beef-0000-000000000000", "prompt": "test",
    }, timeout=5)
    assert r.status_code == 404
