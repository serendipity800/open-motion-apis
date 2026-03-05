"""MoMask backend server (port 8081).

Run from the momask-codes directory:

    cd /media/data/canzhi_data/baselines/momask-codes
    python ../motion_api/backends/momask_server.py --port 8081 --device cuda:0 --dataset t2m
"""

from __future__ import annotations

import argparse
import sys
import os

import numpy as np
import torch
import uvicorn

# ---------------------------------------------------------------------------
# Ensure we can import from motion_api (two directories up from this file)
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MOTION_API_DIR = os.path.dirname(_THIS_DIR)          # baselines/motion_api
_BASELINES_DIR = os.path.dirname(_MOTION_API_DIR)     # baselines/
sys.path.insert(0, _BASELINES_DIR)
# Also add cwd so that model-local imports (models/, utils/, etc.) work
# when the server is launched from the model's own directory.
sys.path.insert(0, os.getcwd())

from motion_api.schemas import GenerateRequest, GenerateResponse, MotionChoice, MotionData
from motion_api.server_base import create_app, make_response_id, now_ts
from motion_api.utils import encode_motion
from motion_api.config import DEFAULT_PARAMS, FPS

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
_model_state: dict = {}


def load_model(opt) -> None:
    """Load all MoMask sub-models into _model_state."""
    # These imports require the momask-codes directory to be on sys.path,
    # which is guaranteed when the server is launched from that directory.
    from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
    from models.vq.model import RVQVAE, LengthEstimator
    from utils.get_opt import get_opt as get_model_opt
    from utils.motion_process import recover_from_ric
    from os.path import join as pjoin

    clip_version = "ViT-B/32"
    dim_pose = 251 if opt.dataset == "kit" else 263
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset, opt.name)
    model_opt_path = pjoin(root_dir, "opt.txt")
    model_opt = get_model_opt(model_opt_path, device=opt.device)

    # --- VQ model ---
    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset, model_opt.vq_name, "opt.txt")
    vq_opt = get_model_opt(vq_opt_path, device=opt.device)
    vq_opt.dim_pose = dim_pose

    vq_model = RVQVAE(
        vq_opt,
        vq_opt.dim_pose,
        vq_opt.nb_code,
        vq_opt.code_dim,
        vq_opt.output_emb_width,
        vq_opt.down_t,
        vq_opt.stride_t,
        vq_opt.width,
        vq_opt.depth,
        vq_opt.dilation_growth_rate,
        vq_opt.vq_act,
        vq_opt.vq_norm,
    )
    ckpt = torch.load(
        pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, "model", "net_best_fid.tar"),
        map_location="cpu",
    )
    model_key = "vq_model" if "vq_model" in ckpt else "net"
    vq_model.load_state_dict(ckpt[model_key])
    vq_model.eval().to(opt.device)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    # --- Mask Transformer ---
    t2m_transformer = MaskTransformer(
        code_dim=model_opt.code_dim,
        cond_mode="text",
        latent_dim=model_opt.latent_dim,
        ff_size=model_opt.ff_size,
        num_layers=model_opt.n_layers,
        num_heads=model_opt.n_heads,
        dropout=model_opt.dropout,
        clip_dim=512,
        cond_drop_prob=model_opt.cond_drop_prob,
        clip_version=clip_version,
        opt=model_opt,
    )
    ckpt = torch.load(
        pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, "model", "latest.tar"),
        map_location="cpu",
    )
    model_key = "t2m_transformer" if "t2m_transformer" in ckpt else "trans"
    missing, unexpected = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected) == 0
    assert all(k.startswith("clip_model.") for k in missing)
    t2m_transformer.eval().to(opt.device)

    # --- Residual Transformer ---
    res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset, opt.res_name, "opt.txt")
    res_opt = get_model_opt(res_opt_path, device=opt.device)
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens = vq_opt.nb_code

    res_transformer = ResidualTransformer(
        code_dim=vq_opt.code_dim,
        cond_mode="text",
        latent_dim=res_opt.latent_dim,
        ff_size=res_opt.ff_size,
        num_layers=res_opt.n_layers,
        num_heads=res_opt.n_heads,
        dropout=res_opt.dropout,
        clip_dim=512,
        shared_codebook=vq_opt.shared_codebook,
        cond_drop_prob=res_opt.cond_drop_prob,
        share_weight=res_opt.share_weight,
        clip_version=clip_version,
        opt=res_opt,
    )
    ckpt = torch.load(
        pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, "model", "net_best_fid.tar"),
        map_location=opt.device,
    )
    missing, unexpected = res_transformer.load_state_dict(ckpt["res_transformer"], strict=False)
    assert len(unexpected) == 0
    assert all(k.startswith("clip_model.") for k in missing)
    res_transformer.eval().to(opt.device)

    # --- Length Estimator ---
    length_estimator = LengthEstimator(512, 50)
    ckpt = torch.load(
        pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, "length_estimator", "model", "finest.tar"),
        map_location=opt.device,
    )
    length_estimator.load_state_dict(ckpt["estimator"])
    length_estimator.eval().to(opt.device)

    # --- Mean / Std for inv_transform ---
    mean = np.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, "meta", "mean.npy"))
    std = np.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, "meta", "std.npy"))

    _model_state.update(
        {
            "vq_model": vq_model,
            "t2m_transformer": t2m_transformer,
            "res_transformer": res_transformer,
            "length_estimator": length_estimator,
            "mean": mean,
            "std": std,
            "recover_from_ric": recover_from_ric,
            "device": opt.device,
            "num_joints": 21 if opt.dataset == "kit" else 22,
        }
    )
    print("MoMask models loaded successfully.")


def generate(req: GenerateRequest) -> GenerateResponse:
    from torch.distributions.categorical import Categorical
    import torch.nn.functional as F

    ms = _model_state
    device = ms["device"]
    sp = req.sampling_params
    defaults = DEFAULT_PARAMS["momask"]

    cond_scale = sp.cond_scale if sp.cond_scale is not None else defaults["cond_scale"]
    temperature = sp.temperature if sp.temperature is not None else defaults["temperature"]
    topkr = sp.topkr if sp.topkr is not None else defaults["topkr"]
    time_steps = sp.time_steps if sp.time_steps is not None else defaults["time_steps"]
    gumbel_sample = sp.gumbel_sample if sp.gumbel_sample is not None else defaults["gumbel_sample"]

    torch.manual_seed(req.seed)
    np.random.seed(req.seed)

    num_joints = ms["num_joints"]
    captions = [req.prompt] * req.num_samples

    # Estimate motion length in tokens
    fps = FPS
    n_frames = int(req.motion_length * fps)
    token_lens = torch.LongTensor([n_frames // 4] * req.num_samples).to(device)
    m_lengths = token_lens * 4

    with torch.no_grad():
        mids = ms["t2m_transformer"].generate(
            captions,
            token_lens,
            timesteps=time_steps,
            cond_scale=cond_scale,
            temperature=temperature,
            topk_filter_thres=topkr,
            gsample=gumbel_sample,
        )
        mids = ms["res_transformer"].generate(mids, captions, token_lens, temperature=1, cond_scale=5)
        pred_motions = ms["vq_model"].forward_decoder(mids)
        pred_motions = pred_motions.detach().cpu().numpy()

    inv_motions = pred_motions * ms["std"] + ms["mean"]

    choices = []
    for i in range(req.num_samples):
        length = int(m_lengths[i])
        joint_data = inv_motions[i][:length]  # (T, 263)
        joint_tensor = torch.from_numpy(joint_data).float()
        joints = ms["recover_from_ric"](joint_tensor, num_joints).numpy()  # (T, 22, 3)

        choices.append(
            MotionChoice(
                index=i,
                motion=MotionData(
                    num_frames=length,
                    num_joints=num_joints,
                    fps=fps,
                    data=encode_motion(joints),
                ),
            )
        )

    total_frames = sum(c.motion.num_frames for c in choices)
    return GenerateResponse(
        id=make_response_id(),
        model="momask",
        prompt=req.prompt,
        created=now_ts(),
        choices=choices,
        usage={"generation_frames": total_frames},
    )


def parse_args():
    parser = argparse.ArgumentParser(description="MoMask API Server")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset", type=str, default="t2m", choices=["t2m", "kit"])
    parser.add_argument("--checkpoints-dir", type=str, default="./checkpoints")
    parser.add_argument("--name", type=str, default="t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns",
                        help="Mask Transformer model name")
    parser.add_argument("--res-name", type=str, default="tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw",
                        help="Residual Transformer model name")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Build a simple namespace that load_model expects
    class Opt:
        pass

    opt = Opt()
    opt.device = torch.device(args.device)
    opt.dataset = args.dataset
    opt.checkpoints_dir = args.checkpoints_dir
    opt.name = args.name
    opt.res_name = args.res_name

    load_model(opt)
    app = create_app("momask", generate)
    uvicorn.run(app, host=args.host, port=args.port)
