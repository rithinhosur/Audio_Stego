"""Audio quality and basic forensic metrics."""
from __future__ import annotations
import numpy as np

def mse(cover, stego) -> float:
    a, b = np.asarray(cover, dtype=np.float64), np.asarray(stego, dtype=np.float64)
    if a.shape != b.shape: raise ValueError("Audio arrays must have identical shape.")
    return float(np.mean((a-b)**2)) if a.size else 0.0

def psnr(cover, stego) -> float:
    err = mse(cover, stego)
    return float("inf") if err == 0 else float(20*np.log10(32767.0)-10*np.log10(err))

def snr(cover, stego) -> float:
    a = np.asarray(cover, dtype=np.float64); noise = a-np.asarray(stego, dtype=np.float64)
    p = float(np.mean(a*a)) if a.size else 0.0
    n = float(np.mean(noise*noise)) if noise.size else 0.0
    return float("inf") if n == 0 else (float("-inf") if p == 0 else float(10*np.log10(p/n)))

def segmental_snr(cover, stego, frame_len: int = 256) -> float:
    a = np.asarray(cover, dtype=np.float64).reshape(-1); b = np.asarray(stego, dtype=np.float64).reshape(-1)
    vals=[]
    for i in range(0, min(len(a),len(b)), frame_len):
        x=a[i:i+frame_len]; y=b[i:i+frame_len]
        if not len(x): continue
        signal=float(np.mean(x*x)); noise=float(np.mean((x-y)**2))
        if noise == 0: vals.append(100.0)
        else: vals.append(float(np.clip(10*np.log10(max(signal,1e-12)/noise), -10, 35)))
    return float(np.mean(vals)) if vals else 0.0

def histogram_error_ratio(cover, stego) -> float:
    a=np.asarray(cover).ravel(); b=np.asarray(stego).ravel()
    ha=np.bincount(a.astype(np.int32)+32768, minlength=65536).astype(np.float64)
    hb=np.bincount(b.astype(np.int32)+32768, minlength=65536).astype(np.float64)
    denom=float(ha.sum())
    return float(np.abs(ha-hb).sum()/denom) if denom else 0.0

def moments_dr(cover, stego) -> dict:
    def moments(x):
        x=np.asarray(x,dtype=np.float64).ravel()
        if not x.size: return np.zeros(4)
        m=x.mean(); centered=x-m; v=float(np.mean(centered**2))
        skew=float(np.mean(centered**3)/(v**1.5)) if v>0 else 0.0
        kurt=float(np.mean(centered**4)/(v**2)-3) if v>0 else 0.0
        return np.array([m,v,skew,kurt])
    a,b=moments(cover),moments(stego)
    return {name:float(abs(y-x)/max(abs(x),1e-12)*100) for name,x,y in zip(("mean","variance","skewness","kurtosis"),a,b)}
