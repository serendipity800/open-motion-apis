"""MLD backend server (port 8083).

Run from the motion-latent-diffusion directory:

    cd /media/data/canzhi_data/baselines/motion-latent-diffusion
    python ../motion_api/backends/mld_server.py \\
        --port 8083 --device cuda:0 \\
        --cfg ./configs/config_mld_humanml3d.yaml
"""

from __future__ import annotations

import argparse
import sys
import os

import numpy as np
import torch
import uvicorn

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MOTION_API_DIR = os.path.dirname(_THIS_DIR)
_BASELINES_DIR = os.path.dirname(_MOTION_API_DIR)
sys.path.insert(0, _BASELINES_DIR)
sys.path.insert(0, os.getcwd())

from motion_api.schemas import GenerateRequest, GenerateResponse, MotionChoice, MotionData
from motion_api.server_base import create_app, make_response_id, now_ts
from motion_api.utils import encode_motion
from motion_api.config import DEFAULT_PARAMS, FPS

_model_state: dict = {}


def load_model(cfg_path: str, device_str: str, sample_mean: bool = False) -> None:
    """Load the MLD model."""
    from mld.config import parse_args as mld_parse_args
    from mld.data.get_data import get_datasets
    from mld.models.get_model import get_model
    from mld.utils.logger import create_logger

    # MLD reads config from YAML; monkeypatch sys.argv so parse_args works
    saved_argv = sys.argv
    sys.argv = ["demo.py", "--cfg", cfg_path, "--cfg_assets", "./configs/assets.yaml"]
    cfg = mld_parse_args(phase="demo")
    sys.argv = saved_argv

    cfg.FOLDER = cfg.TEST.FOLDER

    device = torch.device(device_str)

    # Load dataset (needed to build model)
    dataset = get_datasets(cfg, phase="test")[0]
    model = get_model(cfg, dataset)

    # Load checkpoint
    state_dict = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=True)

    model.sample_mean = cfg.TEST.MEAN if not sample_mean else True
    model.fact = cfg.TEST.FACT
    model.to(device)
    model.eval()

    _model_state.update(
        {
            "model": model,
            "device": device,
            "cfg": cfg,
        }
    )
    print("MLD model loaded successfully.")


def generate(req: GenerateRequest) -> GenerateResponse:
    ms = _model_state
    sp = req.sampling_params
    defaults = DEFAULT_PARAMS["mld"]

    sample_mean = sp.sample_mean if sp.sample_mean is not None else defaults["sample_mean"]

    fps = FPS
    n_frames = int(req.motion_length * fps)

    torch.manual_seed(req.seed)
    np.random.seed(req.seed)

    model = ms["model"]
    device = ms["device"]

    # Override sample_mean per request
    model.sample_mean = sample_mean

    texts = [req.prompt] * req.num_samples
    lengths = [n_frames] * req.num_samples
    batch = {"length": lengths, "text": texts}

    with torch.no_grad():
        joints_list = model(batch)  # list of tensors, each (T, 22, 3)

    choices = []
    for i, joints in enumerate(joints_list):
        joints_np = joints.detach().cpu().numpy()  # (T, 22, 3)
        length = joints_np.shape[0]
        choices.append(
            MotionChoice(
                index=i,
                motion=MotionData(
                    num_frames=length,
                    num_joints=joints_np.shape[1],
                    fps=fps,
                    data=encode_motion(joints_np),
                ),
            )
        )

    total_frames = sum(c.motion.num_frames for c in choices)
    return GenerateResponse(
        id=make_response_id(),
        model="mld",
        prompt=req.prompt,
        created=now_ts(),
        choices=choices,
        usage={"generation_frames": total_frames},
    )


def parse_args():
    parser = argparse.ArgumentParser(description="MLD API Server")
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cfg", type=str, required=True,
                        help="Path to MLD YAML config file")
    parser.add_argument("--sample-mean", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_model(args.cfg, args.device, args.sample_mean)
    app = create_app("mld", generate)
    uvicorn.run(app, host=args.host, port=args.port)
