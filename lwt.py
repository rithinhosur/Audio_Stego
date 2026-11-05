"""Channel-aware reversible integer Haar lifting transform."""
from __future__ import annotations
import numpy as np

def forward_lwt(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform interleaved mono/stereo PCM; returns per-channel s, d and odd remainders."""
    x = np.asarray(samples, dtype=np.int64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("Samples must have shape (frames,) or (frames, channels).")
    frames, channels = x.shape
    pairs = frames // 2
    even = x[:2*pairs:2, :]
    odd = x[1:2*pairs:2, :]
    d = odd - even
    s = even + (d >> 1)
    remainder = x[-1:, :].copy() if frames % 2 else np.empty((0, channels), dtype=np.int64)
    return s.astype(np.int64), d.astype(np.int64), remainder.astype(np.int64)

def inverse_lwt(s: np.ndarray, d: np.ndarray, remainder: np.ndarray | None = None) -> np.ndarray:
    """Invert the integer Haar lifting transform exactly when coefficients are unchanged."""
    s = np.asarray(s, dtype=np.int64)
    d = np.asarray(d, dtype=np.int64)
    if s.shape != d.shape or s.ndim != 2:
        raise ValueError("Approximation and detail arrays must have identical 2-D shapes.")
    even = s - (d >> 1)
    odd = even + d
    out = np.empty((s.shape[0]*2, s.shape[1]), dtype=np.int64)
    out[0::2] = even
    out[1::2] = odd
    if remainder is not None and np.asarray(remainder).size:
        rem = np.asarray(remainder, dtype=np.int64).reshape(1, -1)
        out = np.vstack([out, rem])
    return out[:, 0] if out.shape[1] == 1 else out

def roundtrip_error(samples: np.ndarray) -> int:
    s, d, rem = forward_lwt(samples)
    restored = inverse_lwt(s, d, rem)
    return int(np.max(np.abs(np.asarray(samples, dtype=np.int64) - restored))) if np.size(samples) else 0
