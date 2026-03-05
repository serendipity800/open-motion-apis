"""T2M-GPT backend server (port 8084).

Run from the T2M-GPT directory:

    cd /media/data/canzhi_data/baselines/T2M-GPT
    python ../motion_api/backends/t2m_gpt_server.py \\
        --port 8084 --device cuda:0 --dataset t2m \\
        --resume-pth ./checkpoints/t2m/vq/net_best_fid.tar \\
        --resume-trans ./checkpoints/t2m/trans/net_best_fid.tar
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
    """Load VQ-VAE and GPT transformer for T2M-GPT."""
    import clip
    import models.vqvae as vqvae
    import models.t2m_trans as trans
    from utils.motion_process import recover_from_ric

    device = torch.device(args.device)

    # --- CLIP ---
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip.model.convert_weights(clip_model)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    # --- VQ-VAE ---
    net = vqvae.HumanVQVAE(
        args,
        args.nb_code,
        args.code_dim,
        args.output_emb_width,
        args.down_t,
        args.stride_t,
        args.width,
        args.depth,
        args.dilation_growth_rate,
    )
    ckpt = torch.load(args.resume_pth, map_location="cpu")
    net.load_state_dict(ckpt["net"], strict=True)
    net.eval().to(device)

    # --- GPT Transformer ---
    trans_encoder = trans.Text2Motion_Transformer(
        num_vq=args.nb_code,
        embed_dim=args.embed_dim_gpt,
        clip_dim=args.clip_dim,
        block_size=args.block_size,
        num_layers=args.num_layers,
        n_head=args.n_head_gpt,
        drop_out_rate=args.drop_out_rate,
        fc_rate=args.ff_rate,
    )
    if args.resume_trans is not None:
        ckpt = torch.load(args.resume_trans, map_location="cpu")
        trans_encoder.load_state_dict(ckpt["trans"], strict=True)
    trans_encoder.eval().to(device)  # eval mode: evaluation_transformer_test calls trans.eval() internally

    # Mean / Std – must match the normalization used during VQ-VAE training
    # GPT_eval_multi.py / dataset_TM_eval.py load from the VQ-VAE meta dir, NOT HumanML3D/
    dataname = args.dataname
    if dataname == "t2m":
        mean = np.load("./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy")
        std  = np.load("./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy")
    else:
        mean = np.load("./checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy")
        std  = np.load("./checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy")

    _model_state.update(
        {
            "net": net,
            "trans_encoder": trans_encoder,
            "clip_model": clip_model,
            "mean": mean,
            "std": std,
            "recover_from_ric": recover_from_ric,
            "device": device,
            "num_joints": 21 if dataname == "kit" else 22,
            "block_size": args.block_size,
        }
    )
    print("T2M-GPT model loaded successfully.")


def generate(req: GenerateRequest) -> GenerateResponse:
    import clip as clip_lib

    ms = _model_state
    sp = req.sampling_params
    defaults = DEFAULT_PARAMS["t2m_gpt"]

    temperature = sp.temperature if sp.temperature is not None else defaults["temperature"]
    if_categorial = sp.if_categorial if sp.if_categorial is not None else defaults["if_categorial"]

    fps = FPS
    n_frames = int(req.motion_length * fps)
    # T2M-GPT uses tokens of stride 4
    token_len = n_frames // 4
    num_joints = ms["num_joints"]
    device = ms["device"]

    torch.manual_seed(req.seed)
    np.random.seed(req.seed)

    texts = [req.prompt] * req.num_samples

    # Encode text via CLIP
    with torch.no_grad():
        text_tokens = clip_lib.tokenize(texts, truncate=True).to(device)
        clip_feature = ms["clip_model"].encode_text(text_tokens).float()  # (B, 512)

    # Generate token sequences for each sample
    choices = []
    with torch.no_grad():
        for i in range(req.num_samples):
            feat = clip_feature[i : i + 1]  # (1, 512)
            index_motion = ms["trans_encoder"].sample(feat, if_categorial=if_categorial)
            # index_motion: (1, T_token)
            pred_pose = ms["net"].forward_decoder(index_motion)  # (1, T*4, 263)
            pred_pose = pred_pose.detach().cpu().numpy()

            # inv transform
            motion = pred_pose[0] * ms["std"] + ms["mean"]  # (T, 263)
            length = min(motion.shape[0], n_frames)
            motion = motion[:length]

            joint_tensor = torch.from_numpy(motion).float()
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
        model="t2m_gpt",
        prompt=req.prompt,
        created=now_ts(),
        choices=choices,
        usage={"generation_frames": total_frames},
    )


def parse_args():
    parser = argparse.ArgumentParser(description="T2M-GPT API Server")
    parser.add_argument("--port", type=int, default=8084)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset", "--dataname", dest="dataname", type=str, default="t2m",
                        choices=["t2m", "kit"])
    # VQ-VAE args (defaults match T2M-GPT t2m config)
    parser.add_argument("--resume-pth", type=str, required=True, help="VQ-VAE checkpoint")
    parser.add_argument("--resume-trans", type=str, default=None, help="GPT transformer checkpoint")
    parser.add_argument("--nb-code", type=int, default=512)
    parser.add_argument("--code-dim", type=int, default=512)
    parser.add_argument("--output-emb-width", type=int, default=512)
    parser.add_argument("--down-t", type=int, default=3)
    parser.add_argument("--stride-t", type=int, default=2)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dilation-growth-rate", type=int, default=3)
    parser.add_argument("--vq-act", type=str, default="relu")
    parser.add_argument("--quantizer", type=str, default="ema_reset",
                        choices=["ema_reset", "orig", "ema", "reset"])
    parser.add_argument("--mu", type=float, default=0.99)
    parser.add_argument("--vq-norm", type=str, default=None)
    # GPT args
    parser.add_argument("--block-size", type=int, default=51)
    parser.add_argument("--embed-dim-gpt", type=int, default=512)
    parser.add_argument("--clip-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--n-head-gpt", type=int, default=8)
    parser.add_argument("--ff-rate", type=int, default=4)
    parser.add_argument("--drop-out-rate", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_model(args)
    app = create_app("t2m_gpt", generate)
    uvicorn.run(app, host=args.host, port=args.port)
