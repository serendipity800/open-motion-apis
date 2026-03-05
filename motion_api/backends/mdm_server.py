"""MDM backend server (port 8082).

Run from the motion-diffusion-model directory:

    cd /media/data/canzhi_data/baselines/motion-diffusion-model
    python ../motion_api/backends/mdm_server.py \\
        --port 8082 --device cuda:0 \\
        --model-path ./save/humanml_enc_512_50steps/model000200000.pt
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


def load_model(args) -> None:
    """Load the MDM model."""
    import json, argparse as _ap
    from utils.model_util import create_model_and_diffusion, load_saved_model
    from utils import dist_util
    from utils.sampler_util import ClassifierFreeSampleModel
    from data_loaders.humanml.scripts.motion_process import recover_from_ric

    # dist_util expects an integer device index, not "cuda:N"
    dev_str = str(args.device)
    dev_idx = int(dev_str.split(":")[-1]) if ":" in dev_str else int(dev_str)
    dist_util.setup_dist(dev_idx)

    # Load training args from args.json next to the checkpoint, then overlay
    # our server-specific values (device, guidance_param, model_path).
    args_json = os.path.join(os.path.dirname(args.model_path), "args.json")
    if os.path.exists(args_json):
        with open(args_json) as f:
            saved = json.load(f)
        merged = _ap.Namespace(**saved)
        # overlay server args
        merged.model_path = args.model_path
        merged.device = args.device
        merged.guidance_param = getattr(args, "guidance_param", saved.get("gen_guidance_param", 2.5))
        merged.use_ema = getattr(args, "use_ema", saved.get("use_ema", False))
        args = merged

    from data_loaders.get_data import get_dataset_loader
    data = get_dataset_loader(
        name=args.dataset,
        batch_size=1,
        num_frames=196,
        split="test",
        hml_mode="text_only",
    )

    model, diffusion = create_model_and_diffusion(args, data)
    load_saved_model(model, args.model_path, use_avg=getattr(args, "use_ema", False))

    if getattr(args, "guidance_param", 1) != 1:
        model = ClassifierFreeSampleModel(model)

    model.to(dist_util.dev())
    model.eval()

    _model_state.update(
        {
            "model": model,
            "diffusion": diffusion,
            "data": data,
            "recover_from_ric": recover_from_ric,
            "device": args.device,
            "n_joints": 22 if args.dataset == "humanml" else 21,
        }
    )
    print("MDM model loaded successfully.")


def generate(req: GenerateRequest) -> GenerateResponse:
    from data_loaders.tensors import collate

    ms = _model_state
    sp = req.sampling_params
    defaults = DEFAULT_PARAMS["mdm"]

    guidance_param = sp.guidance_param if sp.guidance_param is not None else defaults["guidance_param"]
    num_repetitions = sp.num_repetitions if sp.num_repetitions is not None else defaults["num_repetitions"]

    fps = FPS
    n_frames = min(196, int(req.motion_length * fps))
    n_joints = ms["n_joints"]

    torch.manual_seed(req.seed)
    np.random.seed(req.seed)

    texts = [req.prompt] * req.num_samples
    collate_args = [{"inp": torch.zeros(n_frames), "tokens": None, "lengths": n_frames}] * req.num_samples
    collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
    _, model_kwargs = collate(collate_args)

    from utils import dist_util
    model_kwargs["y"] = {
        k: v.to(dist_util.dev()) if torch.is_tensor(v) else v
        for k, v in model_kwargs["y"].items()
    }

    if guidance_param != 1:
        model_kwargs["y"]["scale"] = torch.ones(req.num_samples, device=dist_util.dev()) * guidance_param

    model = ms["model"]
    diffusion = ms["diffusion"]

    # Encode text once
    if "text" in model_kwargs["y"]:
        model_kwargs["y"]["text_embed"] = model.encode_text(model_kwargs["y"]["text"])

    motion_shape = (req.num_samples, model.njoints, model.nfeats, n_frames)

    all_motions = []
    with torch.no_grad():
        for _ in range(num_repetitions):
            sample = diffusion.p_sample_loop(
                model,
                motion_shape,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )

            # Convert from model output to xyz positions
            if model.data_rep == "hml_vec":
                sample = ms["data"].dataset.t2m_dataset.inv_transform(
                    sample.cpu().permute(0, 2, 3, 1)
                ).float()
                sample = ms["recover_from_ric"](sample, n_joints)
                sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

            rot2xyz_pose_rep = "xyz" if model.data_rep in ["xyz", "hml_vec"] else model.data_rep
            rot2xyz_mask = (
                None
                if rot2xyz_pose_rep == "xyz"
                else model_kwargs["y"]["mask"].reshape(req.num_samples, n_frames).bool()
            )
            sample = model.rot2xyz(
                x=sample,
                mask=rot2xyz_mask,
                pose_rep=rot2xyz_pose_rep,
                glob=True,
                translation=True,
                jointstype="smpl",
                vertstrans=True,
                betas=None,
                beta=0,
                glob_rot=None,
                get_rotations_back=False,
            )
            all_motions.append(sample.cpu().numpy())

    # all_motions: list of (B, n_joints, 3, T) arrays
    # Concatenate along batch dim: (B*rep, n_joints, 3, T) → take first num_samples
    all_motions_np = np.concatenate(all_motions, axis=0)[: req.num_samples]
    lengths = model_kwargs["y"]["lengths"].cpu().numpy()[: req.num_samples]

    choices = []
    for i in range(req.num_samples):
        length = int(lengths[i])
        # shape: (n_joints, 3, T) → (T, n_joints, 3)
        joints = all_motions_np[i].transpose(2, 0, 1)[:length]
        choices.append(
            MotionChoice(
                index=i,
                motion=MotionData(
                    num_frames=length,
                    num_joints=n_joints,
                    fps=fps,
                    data=encode_motion(joints),
                ),
            )
        )

    total_frames = sum(c.motion.num_frames for c in choices)
    return GenerateResponse(
        id=make_response_id(),
        model="mdm",
        prompt=req.prompt,
        created=now_ts(),
        choices=choices,
        usage={"generation_frames": total_frames},
    )


def parse_args():
    parser = argparse.ArgumentParser(description="MDM API Server")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to MDM checkpoint .pt file")
    parser.add_argument("--dataset", type=str, default="humanml")
    parser.add_argument("--guidance-param", type=float, default=2.5)
    parser.add_argument("--use-ema", action="store_true")
    # Pass remaining MDM args through as needed
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Rename hyphenated attrs to match MDM's expected attribute names
    args.model_path = args.model_path
    args.guidance_param = args.guidance_param
    args.use_ema = args.use_ema

    load_model(args)
    app = create_app("mdm", generate)
    uvicorn.run(app, host=args.host, port=args.port)
