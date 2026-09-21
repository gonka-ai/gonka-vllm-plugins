"""PoC artifact validation logic."""
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from vllm.logger import init_logger

from .data import decode_vector, fraud_test, DEFAULT_DIST_THRESHOLD, DEFAULT_P_MISMATCH, DEFAULT_FRAUD_THRESHOLD

logger = init_logger(__name__)

# Per-step vector-divergence tolerance: a decode step "diverged" if its pre-snap
# cosine distance exceeds this. Honest cross-HW jitter sits far below it and fraud
# far above (pilot: ~60x gap), so the exact value is not sensitive — one knob, like
# stat_test.dist_threshold for the discrete channel. Lets the report show the vector
# channel as a plain "% of steps diverged" rate, on the SAME scale as the k rate.
VECTOR_STEP_TOL = float(os.environ.get("VLLM_POC_VECTOR_TOL", "0.01"))


def score_vector_channel(
    computed_artifacts: List[Dict],
    ref_vectors: Dict[int, List[str]],
) -> Optional[Dict]:
    """Continuous vector-channel score: per-step cosine distance between the
    prover's pre-snap sphere slices (``sph_values_steps`` from the reference
    artifacts) and the validator's own teacher-forced recompute.

    The snap keeps ~4 bits/step, so a subtle fraud sits a few pp above the
    honest cross-HW flip floor; the pre-snap slices carry the displacement
    field the snap discards (A100 pilot: fraud/floor 60x, no per-nonce
    overlap). The k-id chain and the k-based verdict are untouched — this is
    evidence, scored by ONE validator.

    Scores decode steps only (index 0 = prefill, which has its own legacy
    vector_b64 path); non-finite slices are skipped like the NaN guard.
    Returns None when no (computed, reference) vector pair exists, so callers
    attach it as optional evidence.

    Dim-adaptive: the two sides may ship different dims/steps (windowed
    poc_vector_artifacts slices are raw, not renormalized) — both are
    truncated to the common leading dims and renormalized before the cosine.
    """
    per_nonce: List[Dict] = []
    n_skipped = 0      # nonces with a ref but nothing scorable (all-bad/short)
    n_bad_total = 0
    for a in computed_artifacts:
        ref_b64 = ref_vectors.get(a["nonce"]) or []
        own_b64 = a.get("sph_values_steps") or []
        n = min(len(ref_b64), len(own_b64))
        if n < 2:      # need at least one decode step beyond the prefill slice
            n_skipped += bool(ref_b64)
            continue
        dists = []
        n_bad = 0
        for t in range(1, n):
            vp = decode_vector(ref_b64[t])
            vv = decode_vector(own_b64[t])
            d = min(vp.shape[0], vv.shape[0])
            vp, vv = vp[:d], vv[:d]
            if d == 0 or not (
                    np.all(np.isfinite(vp)) and np.all(np.isfinite(vv))):
                n_bad += 1
                continue
            np_norm = float(np.linalg.norm(vp))
            nv_norm = float(np.linalg.norm(vv))
            if np_norm == 0.0 or nv_norm == 0.0:
                n_bad += 1
                continue
            dists.append(1.0 - float(np.dot(vp, vv)) / (np_norm * nv_norm))
        n_bad_total += n_bad
        if dists:
            n_diverged = sum(1 for d in dists if d > VECTOR_STEP_TOL)
            per_nonce.append({
                "nonce": a["nonce"],
                "mean_dist": float(np.mean(dists)),
                "n_diverged": n_diverged,        # steps whose pre-snap vector diverged (> tol)
                "n_steps": len(dists),           # scored decode steps (parallel to the k channel)
                "n_steps_scored": len(dists),    # kept for back-compat
                "n_bad_steps": n_bad,
            })
        else:
            n_skipped += 1     # every step bad — adversarial NaN/zero slices land here
    if not per_nonce:
        return None
    return {
        "mean_dist": float(np.mean([e["mean_dist"] for e in per_nonce])),
        "max_nonce_dist": float(max(e["mean_dist"] for e in per_nonce)),
        # plain, non-expert summary: average "% of steps diverged" per nonce (same
        # scale as the k-mismatch rate — this is what the community report charts).
        "diverged_rate": float(np.mean(
            [e["n_diverged"] / max(e["n_steps"], 1) * 100.0 for e in per_nonce])),
        "step_tol": VECTOR_STEP_TOL,
        "n_nonces_scored": len(per_nonce),
        "n_nonces_skipped": n_skipped,
        "n_bad_steps_total": n_bad_total,
        "per_nonce": per_nonce,
    }


def validate_artifacts(
    computed_artifacts: List[Dict],
    validation_map: Dict[int, str],
    dist_threshold: float = DEFAULT_DIST_THRESHOLD,
    k_dim: int = 12,
) -> Tuple[int, List[int]]:
    """Compare computed artifacts against validation artifacts.

    Args:
        computed_artifacts: List of {"nonce": int, "vector_b64": str}
        validation_map: Dict mapping nonce -> vector_b64
        dist_threshold: L2 distance threshold for mismatch
        k_dim: Expected vector dimension

    Returns:
        (n_mismatch, mismatch_nonces)
    """
    n_mismatch = 0
    mismatch_nonces = []

    for artifact in computed_artifacts:
        nonce = artifact["nonce"]
        received_b64 = validation_map.get(nonce)
        if not received_b64:
            continue

        computed_vec = decode_vector(artifact["vector_b64"])
        received_vec = decode_vector(received_b64)

        if received_vec.shape != (k_dim,):
            n_mismatch += 1
            mismatch_nonces.append(nonce)
            continue

        if not np.all(np.isfinite(received_vec)):
            n_mismatch += 1
            mismatch_nonces.append(nonce)
            continue

        distance = float(np.linalg.norm(computed_vec - received_vec))

        if distance > dist_threshold:
            n_mismatch += 1
            mismatch_nonces.append(nonce)
    
    return n_mismatch, mismatch_nonces


def run_validation(
    computed_artifacts: List[Dict],
    validation_map: Dict[int, str],
    n_total: int,
    dist_threshold: float = DEFAULT_DIST_THRESHOLD,
    p_mismatch: float = DEFAULT_P_MISMATCH,
    fraud_threshold: float = DEFAULT_FRAUD_THRESHOLD,
    k_dim: int = 12,
    use_trajectory: bool = False,
    ref_vectors: Optional[Dict[int, List[str]]] = None,
) -> Dict:
    """Run full validation with fraud test. Same response shape for both flows.

    - prefill (use_trajectory=False): vector-L2 per nonce + binomial fraud_test
      (uses p_mismatch + fraud_threshold). Unchanged.
    - decode (use_trajectory=True, max_tokens>0): a nonce mismatches when the
      validator disagreed with the reference on some step with a snap margin
      above dist_threshold (tau); then the same binomial fraud_test over nonces.
    - ref_vectors (optional, decode): prover-side sph_values_steps per nonce.
      When both sides carry pre-snap slices, the continuous vector-channel score
      (score_vector_channel) is attached as ``vector_score`` EVIDENCE — the
      verdict stays k-based so the two channels can be A/B'd on the same run.
    """
    per_nonce: List[Dict] = []   # per-nonce evidence
    if use_trajectory:
        # One nonce is one trial, exactly as in the prefill flow. The nonce's
        # distance is the largest snap margin among the steps where the
        # validator disagreed with the reference (0.0 when it agreed
        # everywhere); dist_threshold is the margin below which a disagreement
        # is boundary jitter rather than a different computation. The same
        # binomial test then runs over nonces with p_mismatch/fraud_threshold.
        n_mismatch = 0
        mismatch_nonces = []
        for a in computed_artifacts:
            traj = a.get("k_points_steps") or []
            if not traj:
                continue
            m = a.get("n_sphere_mismatches")
            if m is None or m < 0:
                # -1 == no reference was teacher-forced, so nothing was
                # compared. Counting it as zero mismatches would clear any
                # prover whose trajectory the validator never looked at.
                raise ValueError(
                    f"nonce {a['nonce']}: no reference trajectory was compared")
            d = float(a.get("mismatch_margin_max") or 0.0)
            flagged = m > 0 and d > dist_threshold
            n_mismatch += int(flagged)
            per_nonce.append({"nonce": a["nonce"], "n_sphere_mismatches": m,
                              "n_steps": len(traj), "mismatch_margin_max": d,
                              "mismatch": flagged})
            if flagged:
                mismatch_nonces.append(a["nonce"])
        p_value, fraud_detected = fraud_test(n_mismatch, n_total, p_mismatch, fraud_threshold)
    else:
        n_mismatch, mismatch_nonces = validate_artifacts(
            computed_artifacts, validation_map, dist_threshold, k_dim
        )
        p_value, fraud_detected = fraud_test(n_mismatch, n_total, p_mismatch, fraud_threshold)

    result = {
        "n_total": n_total,
        "n_mismatch": n_mismatch,
        "mismatch_nonces": mismatch_nonces,
        "per_nonce": per_nonce,
        "p_value": p_value,
        "fraud_detected": fraud_detected,
    }
    if use_trajectory and ref_vectors:
        vector_score = score_vector_channel(computed_artifacts, ref_vectors)
        if vector_score is not None:
            result["vector_score"] = vector_score
        else:
            # never silent: an adversary shipping all-bad slices (or a
            # misconfigured pair) must not look like "channel not requested"
            logger.warning(
                "vector channel: %d reference trajectories supplied but no "
                "(computed, reference) pair scored", len(ref_vectors))
    return result
