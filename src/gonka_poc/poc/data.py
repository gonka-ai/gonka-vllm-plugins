"""PoC data types and helpers for artifact-based validation."""
import base64
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from scipy.stats import binomtest


# Default validation parameters
DEFAULT_DIST_THRESHOLD = 0.02
DEFAULT_P_MISMATCH = 0.001
DEFAULT_FRAUD_THRESHOLD = 0.01
# Decode: the snap margin above which a teacher-forced disagreement counts as a
# different computation rather than boundary jitter. Sent by the chain as
# stat_test.dist_threshold for decode models; this is only the fallback.
DEFAULT_MARGIN_TAU = 0.025


@dataclass
class Artifact:
    """Single nonce artifact. Prefill PoC carries vector_b64; decode PoC carries the
    sphere_k trajectory (k_points_steps) and leaves vector_b64 empty."""
    nonce: int
    vector_b64: str
    k_points_steps: Optional[List[int]] = None
    # windowed pre-snap slices (poc_vector_artifacts) or full debug trajectory
    sph_values_steps: Optional[List[str]] = None




def encode_vector(vector: np.ndarray) -> str:
    """Encode FP32 vector to base64 FP16 little-endian."""
    f16 = vector.astype('<f2')  # '<f2' = little-endian float16
    return base64.b64encode(f16.tobytes()).decode('ascii')


def decode_vector(b64: str) -> np.ndarray:
    """Decode base64 FP16 little-endian to FP32."""
    data = base64.b64decode(b64)
    f16 = np.frombuffer(data, dtype='<f2')
    return f16.astype(np.float32)




def fraud_test(
    n_mismatch: int,
    n_total: int,
    p_mismatch: float = DEFAULT_P_MISMATCH,
    fraud_threshold: float = DEFAULT_FRAUD_THRESHOLD,
) -> Tuple[float, bool]:
    """
    Run binomial test for fraud detection.
    
    Args:
        n_mismatch: Number of nonces where vectors differ beyond threshold
        n_total: Total nonces checked
        p_mismatch: Expected mismatch rate for honest nodes (baseline)
        fraud_threshold: p-value below which fraud is detected
    
    Returns:
        (p_value, fraud_detected)
    """
    if n_total == 0:
        return 1.0, False

    result = binomtest(
        k=n_mismatch,
        n=n_total,
        p=p_mismatch,
        alternative='greater'
    )
    p_value = float(result.pvalue)
    fraud_detected = p_value < fraud_threshold
    return p_value, fraud_detected

