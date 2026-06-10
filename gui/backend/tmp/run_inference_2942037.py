import sys
sys.path.insert(0, r'/Users/Hassaan/PycharmProjects/KAIR')
import main_test_swinir_config as m

m.CONFIG = {
    "model_path": r'superresolution/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4/models/205000_E.pth',
    "lr_dir": r'testsets/UCMerced_LandUse/HR',
    "hr_dir": r'testsets/UCMerced_LandUse/LR_flat_x2',
    "sr_dir": r'testsets/UCMerced_LandUse/SR',
    "tile": None,
    "tile_overlap": 32,
    "overwrite_sr": True,
    "log_dir": r'testsets/UCMerced_LandUse',
}
m.MODEL_CONFIG = {
    "upscale": 4,
    "in_chans": 3,
    "img_size": 48,
    "window_size": 8,
    "img_range": 1.0,
    "depths": [6, 6, 6, 6, 6, 6],
    "embed_dim": 180,
    "num_heads": [6, 6, 6, 6, 6, 6],
    "mlp_ratio": 2,
    "upsampler": 'pixelshuffle',
    "resi_connection": '1conv',
}
m.main()