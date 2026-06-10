"""
Pydantic schemas for inference endpoints.
"""
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class ModelConfig(BaseModel):
    upscale: int = 2
    in_chans: int = 3
    img_size: int = 128
    window_size: int = 8
    img_range: float = 1.0
    depths: List[int] = Field(default_factory=lambda: [6, 6, 6, 6, 6, 6])
    embed_dim: int = 180
    num_heads: List[int] = Field(default_factory=lambda: [6, 6, 6, 6, 6, 6])
    mlp_ratio: int = 2
    upsampler: str = "pixelshuffle"
    resi_connection: str = "1conv"


class StartInferenceRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model_path: str
    lr_dir: str
    hr_dir: str
    sr_dir: str
    tile: Optional[int] = None
    tile_overlap: int = 32
    overwrite_sr: bool = True
    log_dir: str
    model_network_config: ModelConfig = Field(default_factory=ModelConfig, alias="model_config")


class JobResponse(BaseModel):
    job_id: str
    status: str
