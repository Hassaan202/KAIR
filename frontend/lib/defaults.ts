import type {
  SimplePreprocessConfig, CompletePipelineConfig,
  TrainingJobConfig, InferenceConfig,
  BsrganParams, RealEsrganParams, BsrganPlusParams, SatelliteParams,
} from "./types";

export const DEFAULT_BSRGAN: BsrganParams = {
  jpeg_prob: 0.9, scale2_prob: 0.25, isp_prob: 0.25,
  noise_level1: 2, noise_level2: 25,
};

export const DEFAULT_REAL_ESRGAN: RealEsrganParams = {
  blur_prob_1: 1.0, resize_prob_1: 1.0, gaussian_noise_prob_1: 0.5,
  poisson_noise_prob_1: 0.1, speckle_noise_prob_1: 0.1,
  jpeg_prob_1: 0.9, noise_level1_s1: 2, noise_level2_s1: 25,
  blur_prob_2: 0.8, resize_prob_2: 1.0, gaussian_noise_prob_2: 0.5,
  poisson_noise_prob_2: 0.1, speckle_noise_prob_2: 0.1,
  jpeg_prob_2: 0.8, noise_level1_s2: 2, noise_level2_s2: 15,
  final_jpeg_prob: 0.5, resize_back_prob: 0.5, isp_prob: 0.1,
};

export const DEFAULT_BSRGAN_PLUS: BsrganPlusParams = {
  shuffle_prob: 0.5, use_sharp: false, sharpening_weight: 0.5,
  sharpening_radius: 50, sharpening_threshold: 10,
  poisson_prob: 0.1, speckle_prob: 0.1, isp_prob: 0.1,
  noise_level1: 2, noise_level2: 25,
};

export const DEFAULT_SATELLITE: SatelliteParams = {
  blur_prob_1: 1.0, blur_type_1: "mtf", resize_prob_1: 0.75,
  poisson_prob_1: 0.75, read_noise_prob_1: 0.55, haze_prob_1: 0.45, jpeg_prob_1: 0.12,
  blur_prob_2: 0.92, blur_type_2: "mtf", resize_prob_2: 0.70,
  poisson_prob_2: 0.60, read_noise_prob_2: 0.45, haze_prob_2: 0.35, jpeg_prob_2: 0.08,
  final_jpeg_prob: 0.10, resize_back_prob: 0.35, isp_prob: 0.08,
  noise_level1: 0.8, noise_level2: 5.0,
  mtf_sigma_optics_range: [0.8, 2.8],
  mtf_detector_width_range: [0.7, 1.8],
  mtf_atm_sigma_range: [0.4, 1.8],
};

export const DEFAULT_SIMPLE_PREPROCESS: SimplePreprocessConfig = {
  task: "preprocess_sr_x4",
  pipeline_mode: "hr_only",
  degradation_type: "satellite",
  scale: 4,
  n_channels: 3,
  seed: 42,
  num_workers: 4,
  input_hr_dir: "trainsets/satellite/HR",
  input_lr_dir: null,
  output_hr_dir: "trainsets/satellite/HR_processed",
  output_lr_dir: "trainsets/satellite/LR_processed",
  supported_extensions: [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"],
  save_format: "png",
  save_hr_copy: true,
  normalize_enabled: false,
  normalize_low_percentile: 2,
  normalize_high_percentile: 98,
  cloud_mask_enabled: false,
  cloud_mask_threshold: 0.4,
  cloud_mask_average_over: 4,
  cloud_mask_dilation_size: 2,
  cloud_mask_nodata: 0.0,
  cloud_mask_auto_scale: true,
  bsrgan: DEFAULT_BSRGAN,
  real_esrgan: DEFAULT_REAL_ESRGAN,
  bsrgan_plus: DEFAULT_BSRGAN_PLUS,
  satellite: DEFAULT_SATELLITE,
};

export const DEFAULT_COMPLETE_PIPELINE: CompletePipelineConfig = {
  task: "l2_paired_sr_preprocessing",
  n_channels: 3,
  num_workers: 4,
  seed: null,
  supported_extensions: [".png", ".tif", ".tiff", ".jpg"],
  input_hr_dir: "trainsets/satellite/HR",
  input_lr_dir: "trainsets/satellite/LR",
  input_qa_dir: null,
  output_hr_dir: "trainsets/satellite/HR_out",
  output_lr_dir: "trainsets/satellite/LR_out",
  save_format: "png",
  masking: {
    enabled: true, method: "qa_band",
    invalid_classes: [0, 1, 3, 8, 9, 10, 11],
    s2_threshold: 0.4, s2_average_over: 4,
    s2_dilation_size: 2, s2_auto_scale: true, s2_nodata: 0.0,
  },
  relative_normalization: { enabled: false, method: "histogram_match", direction: "lr_to_hr", mask_aware: true },
  normalization: { enabled: true, low_percentile: 2, high_percentile: 98, mask_aware: true },
  registration: { enabled: true, warp_mode: "translation", num_iters: 50, eps: 1e-5, gauss_filt_size: 5, skip_on_failure: true },
  degradation: {
    enabled: false, type: "satellite", scale: 4, seed: null,
    satellite: DEFAULT_SATELLITE, real_esrgan: DEFAULT_REAL_ESRGAN,
    bsrgan: DEFAULT_BSRGAN, bsrgan_plus: DEFAULT_BSRGAN_PLUS,
  },
  tiling: { enabled: true, crop_size: 256, step: 192, max_invalid_ratio: 0.1, save_format: "png" },
};

export const DEFAULT_TRAINING: TrainingJobConfig = {
  task: "swinir_sr_x4_psnr",
  model: "plain",
  gpu_ids: [0],
  dist: false,
  scale: 4,
  n_channels: 3,
  path: { root: "superresolution", pretrained_netG: null, pretrained_netD: null, pretrained_netE: null },
  datasets: {
    train: {
      name: "train_dataset", dataset_type: "sr",
      dataroot_H: "trainsets/satellite/HR",
      dataroot_L: "trainsets/satellite/LR",
      H_size: 256, use_photometric_aug: false,
      dataloader_shuffle: true, dataloader_num_workers: 4, dataloader_batch_size: 4,
    },
    test: {
      name: "test_dataset", dataset_type: "sr",
      dataroot_H: "testsets/satellite/HR",
      dataroot_L: "testsets/satellite/LR",
      H_size: 256, use_photometric_aug: false,
      dataloader_shuffle: false, dataloader_num_workers: 2, dataloader_batch_size: 1,
    },
  },
  netG: {
    net_type: "swinir", upscale: 4, in_chans: 3,
    img_size: 128, window_size: 8, img_range: 1.0,
    depths: [6, 6, 6, 6, 6, 6], embed_dim: 180,
    num_heads: [6, 6, 6, 6, 6, 6], mlp_ratio: 2,
    upsampler: "pixelshuffle", resi_connection: "1conv", init_type: "default",
  },
  train: {
    G_lossfn_type: "l1", G_lossfn_weight: 1.0,
    F_lossfn_type: "l1", F_lossfn_weight: 1.0,
    F_feature_layer: [2, 7, 16, 25, 34], F_weights: [0.1, 0.1, 1.0, 1.0, 1.0],
    F_use_input_norm: true, F_use_range_norm: false,
    gan_type: "lsgan", D_lossfn_weight: 1.0,
    E_decay: 0.999, D_init_iters: 0,
    G_optimizer_type: "adam", G_optimizer_lr: 1e-4, G_optimizer_wd: 0,
    G_optimizer_betas: [0.9, 0.99], G_optimizer_clipgrad: null, G_optimizer_reuse: true,
    D_optimizer_type: "adam", D_optimizer_lr: 1e-4, D_optimizer_wd: 0, D_optimizer_reuse: false,
    G_scheduler_type: "MultiStepLR",
    G_scheduler_milestones: [300000, 400000, 500000, 600000, 700000],
    G_scheduler_gamma: 0.5,
    D_scheduler_type: "MultiStepLR",
    D_scheduler_milestones: [300000, 400000, 500000, 600000, 700000],
    D_scheduler_gamma: 0.5,
    G_param_strict: true, D_param_strict: true, E_param_strict: true,
    checkpoint_test: 10000, checkpoint_save: 10000, checkpoint_print: 1000,
  },
};

export const DEFAULT_INFERENCE: InferenceConfig = {
  model_path: "superresolution/swinir_sr_x4/models/best_G.pth",
  lr_dir: "testsets/satellite/LR",
  hr_dir: "testsets/satellite/HR",
  sr_dir: "testsets/satellite/SR",
  tile: null,
  tile_overlap: 32,
  overwrite_sr: true,
  log_dir: "testsets/satellite",
  model_config: {
    upscale: 4, in_chans: 3, img_size: 128, window_size: 8, img_range: 1.0,
    depths: [6, 6, 6, 6, 6, 6], embed_dim: 180,
    num_heads: [6, 6, 6, 6, 6, 6], mlp_ratio: 2,
    upsampler: "pixelshuffle", resi_connection: "1conv",
  },
};
