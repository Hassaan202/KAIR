"""Pydantic schemas matching every config param from the repo."""
from __future__ import annotations
from typing import Literal, Optional, List, Tuple, Union
from pydantic import BaseModel, Field


# ---- Degradation model params ----

class BsrganParams(BaseModel):
    jpeg_prob: float = 0.9
    scale2_prob: float = 0.25
    isp_prob: float = 0.25
    noise_level1: float = 2.0
    noise_level2: float = 25.0


class RealEsrganParams(BaseModel):
    blur_prob_1: float = 1.0
    resize_prob_1: float = 1.0
    gaussian_noise_prob_1: float = 0.5
    poisson_noise_prob_1: float = 0.1
    speckle_noise_prob_1: float = 0.1
    jpeg_prob_1: float = 0.9
    noise_level1_s1: float = 2.0
    noise_level2_s1: float = 25.0
    blur_prob_2: float = 0.8
    resize_prob_2: float = 1.0
    gaussian_noise_prob_2: float = 0.5
    poisson_noise_prob_2: float = 0.1
    speckle_noise_prob_2: float = 0.1
    jpeg_prob_2: float = 0.8
    noise_level1_s2: float = 2.0
    noise_level2_s2: float = 15.0
    final_jpeg_prob: float = 0.5
    resize_back_prob: float = 0.5
    isp_prob: float = 0.1


class BsrganPlusParams(BaseModel):
    shuffle_prob: float = 0.5
    use_sharp: bool = False
    sharpening_weight: float = 0.5
    sharpening_radius: int = 50
    sharpening_threshold: float = 10.0
    poisson_prob: float = 0.1
    speckle_prob: float = 0.1
    isp_prob: float = 0.1
    noise_level1: float = 2.0
    noise_level2: float = 25.0


class SatelliteParams(BaseModel):
    blur_prob_1: float = 1.0
    blur_type_1: Literal["mtf", "anisotropic"] = "mtf"
    resize_prob_1: float = 0.75
    poisson_prob_1: float = 0.75
    read_noise_prob_1: float = 0.55
    haze_prob_1: float = 0.45
    jpeg_prob_1: float = 0.12
    blur_prob_2: float = 0.92
    blur_type_2: Literal["mtf", "anisotropic"] = "mtf"
    resize_prob_2: float = 0.70
    poisson_prob_2: float = 0.60
    read_noise_prob_2: float = 0.45
    haze_prob_2: float = 0.35
    jpeg_prob_2: float = 0.08
    final_jpeg_prob: float = 0.10
    resize_back_prob: float = 0.35
    isp_prob: float = 0.08
    noise_level1: float = 0.8
    noise_level2: float = 5.0
    mtf_sigma_optics_range: Tuple[float, float] = (0.8, 2.8)
    mtf_detector_width_range: Tuple[float, float] = (0.7, 1.8)
    mtf_atm_sigma_range: Tuple[float, float] = (0.4, 1.8)


# ---- Simple pipeline (run_pipeline.py) ----

class SimplePreprocessRequest(BaseModel):
    task: str = "preprocess_sr_x4"
    pipeline_mode: Literal["hr_only", "hr_lr_pair"] = "hr_only"
    degradation_type: Literal["bsrgan", "real_esrgan", "bsrgan_plus", "satellite"] = "satellite"
    scale: Literal[2, 3, 4, 8] = 4
    n_channels: Literal[1, 3] = 3
    seed: Optional[int] = 42
    num_workers: int = 4
    input_hr_dir: str
    input_lr_dir: Optional[str] = None
    output_hr_dir: str
    output_lr_dir: str
    supported_extensions: List[str] = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
    save_format: Literal["png", "tif", "jpg"] = "png"
    save_hr_copy: bool = True
    normalize_enabled: bool = False
    normalize_low_percentile: float = 2.0
    normalize_high_percentile: float = 98.0
    cloud_mask_enabled: bool = False
    cloud_mask_threshold: float = 0.4
    cloud_mask_average_over: int = 4
    cloud_mask_dilation_size: int = 2
    cloud_mask_nodata: float = 0.0
    cloud_mask_auto_scale: bool = True
    bsrgan: BsrganParams = Field(default_factory=BsrganParams)
    real_esrgan: RealEsrganParams = Field(default_factory=RealEsrganParams)
    bsrgan_plus: BsrganPlusParams = Field(default_factory=BsrganPlusParams)
    satellite: SatelliteParams = Field(default_factory=SatelliteParams)


# ---- Complete pipeline (complete_pipeline.py) ----

class MaskingConfig(BaseModel):
    enabled: bool = True
    method: Literal["qa_band", "s2cloudless"] = "qa_band"
    invalid_classes: List[int] = [0, 1, 3, 8, 9, 10, 11]
    s2_threshold: float = 0.4
    s2_average_over: int = 4
    s2_dilation_size: int = 2
    s2_auto_scale: bool = True
    s2_nodata: float = 0.0


class RelativeNormConfig(BaseModel):
    enabled: bool = False
    method: Literal["histogram_match", "mean_std_transfer", "none"] = "histogram_match"
    direction: Literal["lr_to_hr", "hr_to_lr"] = "lr_to_hr"
    mask_aware: bool = True


class NormalizationConfig(BaseModel):
    enabled: bool = True
    low_percentile: float = 2.0
    high_percentile: float = 98.0
    mask_aware: bool = True


class RegistrationConfig(BaseModel):
    enabled: bool = True
    warp_mode: Literal["translation", "euclidean", "affine", "homography"] = "translation"
    num_iters: int = 50
    eps: float = 1e-5
    gauss_filt_size: int = 5
    skip_on_failure: bool = True


class DegradationConfig(BaseModel):
    enabled: bool = False
    type: Literal["satellite", "real_esrgan", "bsrgan", "bsrgan_plus"] = "satellite"
    scale: int = 4
    seed: Optional[int] = None
    satellite: SatelliteParams = Field(default_factory=SatelliteParams)
    real_esrgan: RealEsrganParams = Field(default_factory=RealEsrganParams)
    bsrgan: BsrganParams = Field(default_factory=BsrganParams)
    bsrgan_plus: BsrganPlusParams = Field(default_factory=BsrganPlusParams)


class TilingConfig(BaseModel):
    enabled: bool = True
    crop_size: int = 256
    step: int = 192
    max_invalid_ratio: float = 0.1
    save_format: Literal["png", "tif", "jpg"] = "png"


class CompletePipelineRequest(BaseModel):
    task: str = "l2_paired_sr_preprocessing"
    n_channels: Literal[1, 3] = 3
    num_workers: int = 4
    seed: Optional[int] = None
    supported_extensions: List[str] = [".png", ".tif", ".tiff", ".jpg"]
    input_hr_dir: str
    input_lr_dir: Optional[str] = None
    input_qa_dir: Optional[str] = None
    output_hr_dir: str
    output_lr_dir: str
    save_format: Literal["png", "tif", "jpg"] = "png"
    masking: MaskingConfig = Field(default_factory=MaskingConfig)
    relative_normalization: RelativeNormConfig = Field(default_factory=RelativeNormConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    registration: RegistrationConfig = Field(default_factory=RegistrationConfig)
    degradation: DegradationConfig = Field(default_factory=DegradationConfig)
    tiling: TilingConfig = Field(default_factory=TilingConfig)


# ---- Training (main_train_swinir.py / main_train_gan.py) ----

class SwinIRNetG(BaseModel):
    net_type: Literal["swinir"] = "swinir"
    upscale: Literal[2, 3, 4, 8] = 4
    in_chans: Literal[1, 3] = 3
    img_size: Literal[64, 128] = 128
    window_size: Literal[7, 8] = 8
    img_range: float = 1.0
    depths: List[int] = [6, 6, 6, 6, 6, 6]
    embed_dim: Literal[60, 180] = 180
    num_heads: List[int] = [6, 6, 6, 6, 6, 6]
    mlp_ratio: int = 2
    upsampler: Literal["pixelshuffle", "pixelshuffledirect", "nearest+conv"] = "pixelshuffle"
    resi_connection: Literal["1conv", "3conv"] = "1conv"
    init_type: Literal["default", "orthogonal", "normal", "uniform",
                       "xavier_normal", "xavier_uniform", "kaiming_normal", "kaiming_uniform"] = "default"


class NetD(BaseModel):
    net_type: Literal["discriminator_unet", "discriminator_patchgan"] = "discriminator_unet"
    in_nc: int = 3
    base_nc: int = 64
    n_layers: int = 3
    norm_type: Literal["spectral", "batch", "instance", "batchspectral", "instancespectral"] = "spectral"
    init_type: str = "orthogonal"
    init_bn_type: Literal["uniform", "constant"] = "uniform"
    init_gain: float = 0.2


class DatasetConfig(BaseModel):
    name: str = "train_dataset"
    dataset_type: Literal["sr", "blindsr", "dncnn", "fdncnn", "ffdnet",
                          "dnpatch", "srmd", "dpsr", "jpeg", "plain", "plainpatch"] = "sr"
    dataroot_H: str
    dataroot_L: Optional[str] = None
    H_size: int = 256
    use_photometric_aug: bool = False
    dataloader_shuffle: bool = True
    dataloader_num_workers: int = 4
    dataloader_batch_size: int = 4
    degradation_type: Optional[Literal["bsrgan", "bsrgan_plus"]] = None
    shuffle_prob: float = 0.1
    lq_patchsize: int = 64
    use_sharp: bool = False


class TrainSection(BaseModel):
    G_lossfn_type: Literal["l1", "l2", "l2sum", "ssim", "charbonnier"] = "l1"
    G_lossfn_weight: float = 1.0
    F_lossfn_type: Literal["l1", "l2"] = "l1"
    F_lossfn_weight: float = 1.0
    F_feature_layer: Union[int, List[int]] = [2, 7, 16, 25, 34]
    F_weights: Union[float, List[float]] = [0.1, 0.1, 1.0, 1.0, 1.0]
    F_use_input_norm: bool = True
    F_use_range_norm: bool = False
    gan_type: Literal["gan", "ragan", "lsgan", "wgan", "softplusgan"] = "lsgan"
    D_lossfn_weight: float = 1.0
    E_decay: float = 0.999
    D_init_iters: int = 0
    G_optimizer_type: Literal["adam"] = "adam"
    G_optimizer_lr: float = 1e-4
    G_optimizer_wd: float = 0.0
    G_optimizer_betas: Tuple[float, float] = (0.9, 0.99)
    G_optimizer_clipgrad: Optional[float] = None
    G_optimizer_reuse: bool = True
    D_optimizer_type: Literal["adam"] = "adam"
    D_optimizer_lr: float = 1e-4
    D_optimizer_wd: float = 0.0
    D_optimizer_reuse: bool = False
    G_scheduler_type: Literal["MultiStepLR", "CosineAnnealingWarmRestarts"] = "MultiStepLR"
    G_scheduler_milestones: List[int] = [300000, 400000, 500000, 600000, 700000]
    G_scheduler_gamma: float = 0.5
    D_scheduler_type: Literal["MultiStepLR"] = "MultiStepLR"
    D_scheduler_milestones: List[int] = [300000, 400000, 500000, 600000, 700000]
    D_scheduler_gamma: float = 0.5
    G_param_strict: bool = True
    D_param_strict: bool = True
    E_param_strict: bool = True
    checkpoint_test: int = 10000
    checkpoint_save: int = 10000
    checkpoint_print: int = 1000


class TrainingRequest(BaseModel):
    task: str
    model: Literal["plain", "gan"] = "plain"
    gpu_ids: List[int] = [0]
    dist: bool = False
    scale: Literal[2, 3, 4, 8] = 4
    n_channels: Literal[1, 3] = 3
    path: dict = {}
    datasets: dict = {}
    netG: SwinIRNetG = Field(default_factory=SwinIRNetG)
    netD: Optional[NetD] = None
    train: TrainSection = Field(default_factory=TrainSection)


# ---- Inference (main_test_swinir_config.py) ----

class ModelConfig(BaseModel):
    upscale: Literal[2, 3, 4, 8] = 4
    in_chans: Literal[1, 3] = 3
    img_size: Literal[64, 128] = 128
    window_size: Literal[7, 8] = 8
    img_range: float = 1.0
    depths: List[int] = [6, 6, 6, 6, 6, 6]
    embed_dim: Literal[60, 180] = 180
    num_heads: List[int] = [6, 6, 6, 6, 6, 6]
    mlp_ratio: int = 2
    upsampler: Literal["pixelshuffle", "pixelshuffledirect", "nearest+conv"] = "pixelshuffle"
    resi_connection: Literal["1conv", "3conv"] = "1conv"


class InferenceRequest(BaseModel):
    model_path: str
    lr_dir: str
    hr_dir: str
    sr_dir: str
    tile: Optional[int] = None
    tile_overlap: int = 32
    overwrite_sr: bool = True
    log_dir: str
    model_config: ModelConfig = Field(default_factory=ModelConfig)


# ---- Response models ----

class JobResponse(BaseModel):
    job_id: str
    status: Literal["pending", "running", "done", "failed"] = "pending"


class MetricsResult(BaseModel):
    psnr: float
    ssim: float
    it_ssim: float
    sam: float
    uiqi: float
    rmse: float
    fsim: float
    srer: float


class CheckpointInfo(BaseModel):
    name: str
    psnr: float
    path: str
    size_mb: float
    is_best: bool


class GpuStatusResponse(BaseModel):
    gpu_util: float
    vram_used_gb: float
    vram_total_gb: float
    active_job: Optional[dict] = None
