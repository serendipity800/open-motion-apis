from __future__ import annotations

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class SamplingParams(BaseModel):
    """Optional per-model sampling parameters."""
    # MoMask / MDM
    cond_scale: Optional[float] = None
    # MoMask / T2M-GPT
    temperature: Optional[float] = None
    # MoMask
    topkr: Optional[float] = None
    time_steps: Optional[int] = None
    gumbel_sample: Optional[bool] = None
    # MDM
    guidance_param: Optional[float] = None
    num_repetitions: Optional[int] = None
    # MLD
    guidance_scale: Optional[float] = None
    sample_mean: Optional[bool] = None
    # T2M-GPT
    if_categorial: Optional[bool] = None


class GenerateRequest(BaseModel):
    prompt: str
    num_samples: int = 1
    seed: int = 42
    motion_length: float = 6.0  # seconds
    sampling_params: SamplingParams = Field(default_factory=SamplingParams)


class MotionData(BaseModel):
    num_frames: int
    num_joints: int
    fps: int
    data: str  # base64-encoded numpy array, shape (T, 22, 3), float32


class MotionChoice(BaseModel):
    index: int
    motion: MotionData


class GenerateResponse(BaseModel):
    id: str
    model: str
    prompt: str
    created: int
    choices: List[MotionChoice]
    usage: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    model: str


class ModelInfoResponse(BaseModel):
    id: str
    default_params: Dict[str, Any]
