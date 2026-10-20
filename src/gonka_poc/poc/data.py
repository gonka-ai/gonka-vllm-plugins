"""PoC data types and helpers for artifact-based validation."""
import base64
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.stats import binomtest


# Default validation parameters
DEFAULT_DIST_THRESHOLD = 0.02
DEFAULT_P_MISMATCH = 0.001
DEFAULT_FRAUD_THRESHOLD = 0.01


@dataclass
class Artifact:
    """Single nonce artifact with base64-encoded vector."""
    nonce: int
    vector_b64: str


def encode_vector(vector: np.ndarray) -> str:
    """Encode FP32 vector to base64 FP16 little-endian."""
    f16 = vector.astype('<f2')  # '<f2' = little-endian float16
    return base64.b64encode(f16.tobytes()).decode('ascii')


def decode_vector(b64: str) -> np.ndarray:
    """Decode base64 FP16 little-endian to FP32."""
    data = base64.b64decode(b64)
    f16 = np.frombuffer(data, dtype='<f2')
    return f16.astype(np.float32)


def wire_encoding(k_dim: int) -> dict:
    """Wire-protocol encoding descriptor for artifact vectors."""
    return {"dtype": "f16", "k_dim": k_dim, "endian": "le"}


def fraud_test(
    n_mismatch: int,
    n_total: int,
    p_mismatch: float = DEFAULT_P_MISMATCH,
    fraud_threshold: float = DEFAULT_FRAUD_THRESHOLD,
    min_n: int = 0,
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
    if n_total == 0 or n_total < min_n:
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
