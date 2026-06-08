// ---- Preprocessing types (from preprocessing_pipeline/config.json + config_l2.json) ----

export type DegradationType = "bsrgan" | "real_esrgan" | "bsrgan_plus" | "satellite";
export type PipelineMode = "hr_only" | "hr_lr_pair";
export type PipelineVariant = "simple" | "complete";
export type SaveFormat = "png" | "tif" | "jpg";
export type WarpMode = "translation" | "euclidean" | "affine" | "homography";
export type RelNormMethod = "histogram_match" | "mean_std_transfer" | "none";
export type MaskMethod = "qa_band" | "s2cloudless";

export interface BsrganParams {
  jpeg_prob: number;
  scale2_prob: number;
  isp_prob: number;
  noise_level1: number;
  noise_level2: number;
}

export interface RealEsrganParams {
  blur_prob_1: number; resize_prob_1: number;
  gaussian_noise_prob_1: number; poisson_noise_prob_1: number; speckle_noise_prob_1: number;
  jpeg_prob_1: number; noise_level1_s1: number; noise_level2_s1: number;
  blur_prob_2: number; resize_prob_2: number;
  gaussian_noise_prob_2: number; poisson_noise_prob_2: number; speckle_noise_prob_2: number;
  jpeg_prob_2: number; noise_level1_s2: number; noise_level2_s2: number;
  final_jpeg_prob: number; resize_back_prob: number; isp_prob: number;
}

export interface BsrganPlusParams {
  shuffle_prob: number; use_sharp: boolean;
  sharpening_weight: number; sharpening_radius: number; sharpening_threshold: number;
  poisson_prob: number; speckle_prob: number; isp_prob: number;
  noise_level1: number; noise_level2: number;
}

export interface SatelliteParams {
  blur_prob_1: number; blur_type_1: "mtf" | "anisotropic";
  resize_prob_1: number; poisson_prob_1: number; read_noise_prob_1: number;
  haze_prob_1: number; jpeg_prob_1: number;
  blur_prob_2: number; blur_type_2: "mtf" | "anisotropic";
  resize_prob_2: number; poisson_prob_2: number; read_noise_prob_2: number;
  haze_prob_2: number; jpeg_prob_2: number;
  final_jpeg_prob: number; resize_back_prob: number; isp_prob: number;
  noise_level1: number; noise_level2: number;
  mtf_sigma_optics_range: [number, number];
  mtf_detector_width_range: [number, number];
  mtf_atm_sigma_range: [number, number];
}

export interface SimplePreprocessConfig {
  task: string;
  pipeline_mode: PipelineMode;
  degradation_type: DegradationType;
  scale: 2 | 3 | 4 | 8;
  n_channels: 1 | 3;
  seed: number | null;
  num_workers: number;
  input_hr_dir: string;
  input_lr_dir: string | null;
  output_hr_dir: string;
  output_lr_dir: string;
  supported_extensions: string[];
  save_format: SaveFormat;
  save_hr_copy: boolean;
  normalize_enabled: boolean;
  normalize_low_percentile: number;
  normalize_high_percentile: number;
  cloud_mask_enabled: boolean;
  cloud_mask_threshold: number;
  cloud_mask_average_over: number;
  cloud_mask_dilation_size: number;
  cloud_mask_nodata: number;
  cloud_mask_auto_scale: boolean;
  bsrgan: BsrganParams;
  real_esrgan: RealEsrganParams;
  bsrgan_plus: BsrganPlusParams;
  satellite: SatelliteParams;
}

export interface CompletePipelineConfig {
  task: string;
  n_channels: 1 | 3;
  num_workers: number;
  seed: number | null;
  supported_extensions: string[];
  input_hr_dir: string;
  input_lr_dir: string | null;
  input_qa_dir: string | null;
  output_hr_dir: string;
  output_lr_dir: string;
  save_format: SaveFormat;
  masking: {
    enabled: boolean; method: MaskMethod;
    invalid_classes: number[];
    s2_threshold: number; s2_average_over: number;
    s2_dilation_size: number; s2_auto_scale: boolean; s2_nodata: number;
  };
  relative_normalization: {
    enabled: boolean; method: RelNormMethod;
    direction: "lr_to_hr" | "hr_to_lr"; mask_aware: boolean;
  };
  normalization: { enabled: boolean; low_percentile: number; high_percentile: number; mask_aware: boolean; };
  registration: {
    enabled: boolean; warp_mode: WarpMode;
    num_iters: number; eps: number; gauss_filt_size: number; skip_on_failure: boolean;
  };
  degradation: {
    enabled: boolean; type: DegradationType; scale: number; seed: number | null;
    satellite: SatelliteParams; real_esrgan: RealEsrganParams;
    bsrgan: BsrganParams; bsrgan_plus: BsrganPlusParams;
  };
  tiling: { enabled: boolean; crop_size: number; step: number; max_invalid_ratio: number; save_format: SaveFormat; };
}

// ---- Training types (from options JSON files) ----

export type ModelType = "swinir";
export type TrainingMode = "plain" | "gan";
export type LossFn = "l1" | "l2" | "l2sum" | "ssim" | "charbonnier";
export type GanType = "gan" | "ragan" | "lsgan" | "wgan" | "softplusgan";
export type Upsampler = "pixelshuffle" | "pixelshuffledirect" | "nearest+conv";
export type ResiConnection = "1conv" | "3conv";
export type DiscriminatorType = "discriminator_unet" | "discriminator_patchgan";
export type NormType = "spectral" | "batch" | "instance" | "batchspectral" | "instancespectral";
export type InitType = "default" | "orthogonal" | "normal" | "uniform" | "xavier_normal" | "xavier_uniform" | "kaiming_normal" | "kaiming_uniform";
export type DatasetType = "sr" | "blindsr" | "dncnn" | "fdncnn" | "ffdnet" | "dnpatch" | "srmd" | "dpsr" | "jpeg" | "plain" | "plainpatch";
export type SchedulerType = "MultiStepLR" | "CosineAnnealingWarmRestarts";
export type RootDir = "superresolution" | "denoising" | "dejpeg";

export interface SwinIRNetG {
  net_type: "swinir";
  upscale: 2 | 3 | 4 | 8;
  in_chans: 1 | 3;
  img_size: 64 | 128;
  window_size: 7 | 8;
  img_range: 1.0 | 255.0;
  depths: number[];
  embed_dim: 60 | 180;
  num_heads: number[];
  mlp_ratio: number;
  upsampler: Upsampler;
  resi_connection: ResiConnection;
  init_type: InitType;
}

export interface NetD {
  net_type: DiscriminatorType;
  in_nc: number;
  base_nc: number;
  n_layers: number;
  norm_type: NormType;
  init_type: InitType;
  init_bn_type: "uniform" | "constant";
  init_gain: number;
}

export interface TrainDatasetConfig {
  name: string;
  dataset_type: DatasetType;
  dataroot_H: string;
  dataroot_L: string | null;
  H_size: number;
  use_photometric_aug: boolean;
  dataloader_shuffle: boolean;
  dataloader_num_workers: number;
  dataloader_batch_size: number;
  degradation_type?: DegradationType;
  shuffle_prob?: number;
  lq_patchsize?: number;
  use_sharp?: boolean;
}

export interface TrainConfig {
  G_lossfn_type: LossFn;
  G_lossfn_weight: number;
  F_lossfn_type: "l1" | "l2";
  F_lossfn_weight: number;
  F_feature_layer: number | number[];
  F_weights: number | number[];
  F_use_input_norm: boolean;
  F_use_range_norm: boolean;
  gan_type: GanType;
  D_lossfn_weight: number;
  E_decay: number;
  D_init_iters: number;
  G_optimizer_type: "adam";
  G_optimizer_lr: number;
  G_optimizer_wd: number;
  G_optimizer_betas: [number, number];
  G_optimizer_clipgrad: number | null;
  G_optimizer_reuse: boolean;
  D_optimizer_type: "adam";
  D_optimizer_lr: number;
  D_optimizer_wd: number;
  D_optimizer_reuse: boolean;
  G_scheduler_type: SchedulerType;
  G_scheduler_milestones: number[];
  G_scheduler_gamma: number;
  D_scheduler_type: SchedulerType;
  D_scheduler_milestones: number[];
  D_scheduler_gamma: number;
  G_param_strict: boolean;
  D_param_strict: boolean;
  E_param_strict: boolean;
  checkpoint_test: number;
  checkpoint_save: number;
  checkpoint_print: number;
}

export interface TrainingJobConfig {
  task: string;
  model: TrainingMode;
  gpu_ids: number[];
  dist: boolean;
  scale: 2 | 3 | 4 | 8;
  n_channels: 1 | 3;
  path: { root: RootDir; pretrained_netG: string | null; pretrained_netD: string | null; pretrained_netE: string | null; };
  datasets: { train: TrainDatasetConfig; test: TrainDatasetConfig; };
  netG: SwinIRNetG;
  netD?: NetD;
  train: TrainConfig;
}

// ---- Inference types (from main_test_swinir_config.py) ----

export interface InferenceConfig {
  model_path: string;
  lr_dir: string;
  hr_dir: string;
  sr_dir: string;
  tile: number | null;
  tile_overlap: number;
  overwrite_sr: boolean;
  log_dir: string;
  model_config: {
    upscale: 2 | 3 | 4 | 8;
    in_chans: 1 | 3;
    img_size: 64 | 128;
    window_size: 7 | 8;
    img_range: 1.0 | 255.0;
    depths: number[];
    embed_dim: 60 | 180;
    num_heads: number[];
    mlp_ratio: number;
    upsampler: Upsampler;
    resi_connection: ResiConnection;
  };
}

// ---- Metrics (from best_degradation.json + main_test_swinir_config.py) ----

export interface MetricsResult {
  psnr: number;
  ssim: number;
  it_ssim: number;
  sam: number;
  uiqi: number;
  rmse: number;
  fsim: number;
  srer: number;
}

export const METRIC_WEIGHTS: Record<keyof MetricsResult, number> = {
  psnr: 0.20, ssim: 0.20, sam: 0.15, uiqi: 0.10,
  fsim: 0.15, rmse: 0.10, it_ssim: 0.05, srer: 0.05,
};

// ---- Job types ----

export type JobStatus = "pending" | "running" | "done" | "failed";

export interface Job {
  id: string;
  type: "preprocessing" | "training" | "inference";
  status: JobStatus;
  created_at: string;
  progress: number;
  result?: Record<string, unknown>;
  error?: string;
}

export interface GpuStatus {
  gpu_util: number;
  vram_used_gb: number;
  vram_total_gb: number;
  active_job: { id: string; name: string; detail: string; module: string } | null;
}

export interface LogLine {
  ts: string;
  text: string;
  lv: "info" | "ok" | "warn" | "step";
}
