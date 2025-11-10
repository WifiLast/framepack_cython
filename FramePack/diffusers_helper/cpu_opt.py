import os
from typing import Optional

import cv2
import numpy as np

try:  # oneDAL (daal4py) brings vectorized + oneTBB-backed kernels
    import daal4py as d4p  # type: ignore

    _HAS_DAAL = True
except Exception:  # pragma: no cover - optional dependency
    d4p = None
    _HAS_DAAL = False

_CPU_OPT_ENABLED = os.environ.get("FRAMEPACK_CPU_OPT", "1") != "0"
_CPU_OPT_THREADS = os.environ.get("FRAMEPACK_CPU_OPT_THREADS")

if _CPU_OPT_ENABLED:
    try:
        cv2.setUseOptimized(True)
        if _CPU_OPT_THREADS:
            cv2.setNumThreads(max(1, int(_CPU_OPT_THREADS)))
    except Exception:
        pass


def cpu_preprocessing_active() -> bool:
    return _CPU_OPT_ENABLED


def optimized_resize_and_center_crop(image: np.ndarray, target_width: int, target_height: int) -> Optional[np.ndarray]:
    if not _CPU_OPT_ENABLED:
        return None

    if image.shape[0] == target_height and image.shape[1] == target_width:
        return np.ascontiguousarray(image)

    image = np.ascontiguousarray(image)
    original_height, original_width = image.shape[:2]
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = max(1, int(round(original_width * scale_factor)))
    resized_height = max(1, int(round(original_height * scale_factor)))
    interpolation = cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_LANCZOS4
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)

    top = max(0, (resized_height - target_height) // 2)
    left = max(0, (resized_width - target_width) // 2)
    bottom = min(resized_height, top + target_height)
    right = min(resized_width, left + target_width)

    cropped = resized[top:bottom, left:right]
    if cropped.shape[0] != target_height or cropped.shape[1] != target_width:
        padded = np.zeros((target_height, target_width, image.shape[2]), dtype=resized.dtype)
        padded[:cropped.shape[0], :cropped.shape[1]] = cropped
        cropped = padded
    return np.ascontiguousarray(cropped)


def _normalize_with_one_dal(image: np.ndarray) -> Optional[np.ndarray]:
    if not _HAS_DAAL:
        return None
    flat = image.reshape(-1, image.shape[-1]).astype(np.float32, copy=False)
    try:
        algo = d4p.normalization_minmax(flat.shape[1])
        result = algo.compute(flat)
        normalized = result.normalizedData
        if hasattr(normalized, "toArray"):
            normalized = normalized.toArray()
        normalized = np.asarray(normalized, dtype=np.float32)
        normalized = normalized.reshape(image.shape)
        normalized = normalized * 2.0 - 1.0
        return np.ascontiguousarray(normalized)
    except Exception:
        return None


def normalize_uint8_image(image: np.ndarray) -> Optional[np.ndarray]:
    if not _CPU_OPT_ENABLED:
        return None
    normalized = _normalize_with_one_dal(image)
    if normalized is not None:
        return normalized
    normalized = image.astype(np.float32, copy=False)
    normalized = normalized / 127.5 - 1.0
    return np.ascontiguousarray(normalized)
