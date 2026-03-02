import numpy as np


"""
Normalization
"""

# ===========================================================================
# 1. Normalization with percentile clipping
# ===========================================================================
def satellite_pre_norm(img, low_percentile=2, high_percentile=98):
    """Percentile-based normalization for satellite imagery.

    Parameters
    ----------
    img : np.ndarray
        Input image, uint8 or uint16.
    low_percentile : float
        Lower percentile for clipping (removes shadows).  Default: 2.
    high_percentile : float
        Upper percentile for clipping (removes clouds/bright).  Default: 98.

    Returns
    -------
    np.ndarray
        Normalized uint8 image stretched to [0, 255].
    """
    low, high = np.percentile(img, (low_percentile, high_percentile))

    img = np.clip(img, low, high)
    denom = (high - low)
    if denom == 0:
        return img.astype(np.uint8)
    img = (img - low) / denom * 255.0

    return img.astype(np.uint8)



"""
Cloud/Shadow Masking
"""

# ===========================================================================
# 1. Cloud Masking for Sentinel-2 images only
# ===========================================================================
def apply_s2cloudless_mask(
    image_4d: np.ndarray,
    nodata: float = 0.0,
    auto_scale: bool = True,
    threshold: float = 0.4,
    average_over: int = 4,
    dilation_size: int = 2,
) -> np.ndarray:
    """

    Uses the s2cloudless library to detect and mask clouds in Sentinel-2 imagery.
    The detector analyzes 10 spectral bands to identify cloud pixels using a
    machine learning-based approach.
    Sentinel-2 images can be filtered to exclude clouds when retrieving the images from the API

    Requirements:
        - s2cloudless library (pip install s2cloudless)
        - Input images must have exactly 10 bands in the correct order:
          B01, B02, B04, B05, B08, B8A, B09, B10, B11, B12

    Parameters
    ----------
    image_4d : np.ndarray
        Must be shape (1, H, W, 10), float32 or integer.
        Bands MUST be in this exact order:
        B01, B02, B04, B05, B08, B8A, B09, B10, B11, B12
    nodata : float, default 0.0
        Value to set masked pixels to.
    auto_scale : bool, default True
        If True and the array looks like raw Sentinel-2 data
        (integer dtype or max > 1.5), automatically divide by 10_000
        to convert to reflectance [0, 1].
    threshold : float, default 0.4
        Cloud probability threshold (higher = stricter, fewer clouds kept).
    average_over : int, default 4
        Smoothing window size for the cloud detector.
    dilation_size : int, default 2
        Dilation in pixels applied to the cloud mask.

    Returns
    -------
    np.ndarray
        Shape (H, W, 10), same dtype as input (after scaling), with
        cloud pixels set to `nodata`.

    Raises
    ------
    TypeError, ValueError : if input format is wrong.
    ImportError : if s2cloudless is not installed.
    """
    # Lazy import — only needed when cloud masking is actually used
    try:
        from s2cloudless import S2PixelCloudDetector
    except ImportError:
        raise ImportError(
            "s2cloudless is required for cloud masking. "
            "Install it with: pip install s2cloudless"
        )

    # ==================== VALIDATION ====================
    if not isinstance(image_4d, np.ndarray):
        raise TypeError("image_4d must be a numpy ndarray")

    if image_4d.ndim != 4:
        raise ValueError(f"Expected 4D array (1, H, W, 10), got ndim={image_4d.ndim}")

    if image_4d.shape[0] != 1 or image_4d.shape[3] != 10:
        raise ValueError(
            f"Expected shape (1, H, W, 10), got {image_4d.shape}. "
            "First dim must be 1 (batch size), last dim must be 10 bands."
        )

    # Build detector with caller-supplied parameters
    detector = S2PixelCloudDetector(
        threshold=threshold,
        average_over=average_over,
        dilation_size=dilation_size,
    )

    # ==================== PREPROCESSING ====================
    # Work on a copy in float32
    data = image_4d.astype(np.float32, copy=True)

    # Automatic scaling (most common case for raw S2 L1C/L2A)
    if auto_scale:
        max_val = data.max()
        if np.issubdtype(image_4d.dtype, np.integer) or max_val > 1.5:
            data /= 10000.0
            print(f"[s2cloudless] Auto-scaled by /10000 (original max = {max_val:.1f})")
        elif max_val > 1.0:
            raise ValueError(
                "Input values > 1.0 but not integer/raw. "
                "Either set auto_scale=False or manually scale to [0, 1]."
            )

    # Safety clamp (cloud detector expects [0, 1])
    data = np.clip(data, 0.0, 1.0)

    # ==================== MASKING ====================
    cloud_mask = detector.get_cloud_masks(data)          # shape (1, H, W), bool
    mask_expanded = np.broadcast_to(
        cloud_mask[0][..., None], data.shape
    )

    masked = np.where(mask_expanded, nodata, data[0])    # (H, W, 10)

    return masked