import numpy as np

def effective_rank(Z: np.ndarray) -> float:
    """
    Effective rank of a set of representations based on the Shannon entropy 
    of the singular value spectrum.
    
    Args:
        Z : (N, d) — N samples, d dimensions
    Returns:
        erank ∈ [1, d]
    """
    N, d = Z.shape
    Z_c = Z - Z.mean(axis=0)
    
    if N < d:
        _, s, _ = np.linalg.svd(Z_c / np.sqrt(N), full_matrices=False)
        sigma = s ** 2
    else:
        Sigma = (Z_c.T @ Z_c) / N
        sigma = np.linalg.eigvalsh(Sigma)
        
    sigma = sigma[sigma > 1e-10]
    if len(sigma) == 0:
        return 1.0
        
    p = sigma / sigma.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-15))))


def discriminative_ratio(Z_s: np.ndarray, y_s: np.ndarray, eps: float = 1e-8) -> float:
    """
    Fisher's Discriminative Ratio: tr(S_B) / tr(S_W).
    
    Args:
        Z_s : (N_s, d) — Source domain representations
        y_s : (N_s,) — Integer labels {0, ..., C-1}
        eps : Numerical stabilizer
    Returns:
        DR ≥ 0
    """
    N_s = Z_s.shape[0]
    mu = Z_s.mean(axis=0)
    trace_SW = trace_SB = 0.0
    
    for c in np.unique(y_s):
        idx = y_s == c
        Z_c = Z_s[idx]
        n_c = Z_c.shape[0]
        mu_c = Z_c.mean(axis=0)
        
        trace_SW += np.sum((Z_c - mu_c) ** 2) / N_s  # Intra-class (Within)
        diff = mu_c - mu
        trace_SB += (n_c / N_s) * np.dot(diff, diff)  # Inter-class (Between)
        
    return float(trace_SB / (trace_SW + eps))


def ffi(Z_s: np.ndarray, y_s: np.ndarray, d: int = None) -> dict:
    """
    Feature Flattening Index (FFI) — Harmonic mean of normalized erank and DR.
    
    Args:
        Z_s : (N_s, d) — Source domain representations
        y_s : (N_s,) — Integer labels
        d   : Intrinsic dimension (default: Z_s.shape[1])
    Returns:
        Dictionary containing FFI, erank, erank_norm, DR, DR_norm
    """
    if d is None:
        d = Z_s.shape[1]
        
    er = effective_rank(Z_s)
    DR = discriminative_ratio(Z_s, y_s)
    
    en = er / d  # Normalized erank ∈ (0, 1]
    DR_n = DR / (1.0 + DR)  # Normalized DR ∈ [0, 1)
    
    denom = en + DR_n
    val = (2 * en * DR_n / denom) if denom > 1e-10 else 0.0
    
    return {
        "FFI": round(val, 4),
        "erank": round(er, 3),
        "erank_norm": round(en, 4),
        "DR": round(DR, 4),
        "DR_norm": round(DR_n, 4),
    }


def diagnose_feature_flattening(
    Z0_s: np.ndarray, Z0_t: np.ndarray,
    Zs_s: np.ndarray, Zs_t: np.ndarray,
    y_s: np.ndarray,
    tau: float = 0.01) -> dict:
    """
    Comprehensive Feature Flattening diagnosis (Baseline vs. Adapted).
    Evaluates geometric degradation induced by domain alignment.
    """
    d = Z0_s.shape[1]
    r0 = ffi(Z0_s, y_s, d)
    rs = ffi(Zs_s, y_s, d)

    # Linear MMD proxy (Difference of Means)
    gap_0 = float(np.linalg.norm(Z0_s.mean(0) - Z0_t.mean(0)))
    gap_star = float(np.linalg.norm(Zs_s.mean(0) - Zs_t.mean(0)))

    delta_FF = r0["FFI"] - rs["FFI"]
    delta_erank = r0["erank_norm"] - rs["erank_norm"]
    delta_DR = r0["DR_norm"] - rs["DR_norm"]

    alignment_improved = gap_star < gap_0

    # Diagnostic Taxonomy
    if delta_erank > tau and delta_DR > tau:
        cause = "FULL_FLATTENING"
    elif delta_erank > tau:
        cause = "DIMENSIONAL_COLLAPSE"
    elif delta_DR > tau:
        cause = "SEPARABILITY_LOSS"
    else:
        cause = "NONE"

    # Boolean flag strictly tied to the taxonomy output
    flattening_detected = alignment_improved and (cause != "NONE")

    return {
        "flattening_detected": flattening_detected,
        "cause": cause,
        "severity_Delta_FF": round(delta_FF, 4),
        "Delta_erank": round(delta_erank, 4),
        "Delta_DR": round(delta_DR, 4),
        "domain_gap_baseline": round(gap_0, 4),
        "domain_gap_adapted": round(gap_star, 4),
        "baseline": r0,
        "adapted": rs,
    }
