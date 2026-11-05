"""Strict WAV I/O and deterministic synthetic carrier presets."""
from __future__ import annotations
import io, wave
import numpy as np

def read_wav(source) -> tuple[np.ndarray, int]:
    """Read path or bytes as uncompressed 16-bit PCM WAV; return int16 frames and sample rate."""
    close = False
    if isinstance(source, (bytes, bytearray, memoryview)):
        source = io.BytesIO(source); close=True
    try:
        with wave.open(source, "rb") as wf:
            if wf.getsampwidth()!=2 or wf.getcomptype()!="NONE":
                raise ValueError("Only uncompressed 16-bit PCM WAV is supported.")
            channels, rate, frames = wf.getnchannels(), wf.getframerate(), wf.getnframes()
            raw=wf.readframes(frames)
        x=np.frombuffer(raw,dtype="<i2").copy()
        if channels > 1: x=x.reshape(-1,channels)
        return x, rate
    except wave.Error as exc:
        raise ValueError(f"Invalid WAV file: {exc}") from exc

def write_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    x=np.asarray(samples)
    if x.dtype.kind not in "iu" or (x.size and (x.min() < -32768 or x.max() > 32767)):
        raise ValueError("Samples must fit signed 16-bit PCM without clipping.")
    x=x.astype("<i2",copy=False)
    channels=1 if x.ndim==1 else x.shape[1]
    if channels not in (1,2): raise ValueError("Only mono and stereo WAV output is supported.")
    raw=x.reshape(-1).tobytes()
    out=io.BytesIO()
    with wave.open(out,"wb") as wf:
        wf.setnchannels(channels); wf.setsampwidth(2); wf.setframerate(int(sample_rate)); wf.writeframes(raw)
    return out.getvalue()

def synthetic_audio(preset: str, seconds: float=8.0, sample_rate: int=22050, channels: int=1, seed: int=7) -> np.ndarray:
    """Generate reproducible audio-like 16-bit PCM signals for demos, not recorded audio."""
    if seconds <= 0 or sample_rate <= 0 or channels not in (1,2): raise ValueError("Invalid audio settings.")
    n=int(seconds*sample_rate); t=np.arange(n,dtype=np.float64)/sample_rate
    rng=np.random.default_rng(seed)
    if preset=="Speech Formant":
        env=(0.35+0.25*np.sin(2*np.pi*2.2*t))*(0.5+0.5*np.sin(2*np.pi*3.1*t)**2)
        y=env*(0.45*np.sin(2*np.pi*130*t)+0.25*np.sin(2*np.pi*700*t)+0.15*np.sin(2*np.pi*1200*t))
    elif preset=="Classical Harmonic":
        y=0.45*np.sin(2*np.pi*220*t)+0.22*np.sin(2*np.pi*330*t)+0.12*np.sin(2*np.pi*660*t)
    elif preset=="High-Energy Pop":
        y=0.7*np.sin(2*np.pi*110*t)+0.25*np.sin(2*np.pi*440*t)+0.08*rng.normal(size=n)
    elif preset=="Percussive Jazz":
        y=0.2*np.sin(2*np.pi*180*t)+0.18*rng.normal(size=n)
        for start in range(0,n,max(1,sample_rate//3)):
            length=min(sample_rate//20,n-start); y[start:start+length]+=0.65*np.exp(-np.arange(length)/max(1,length/5))*rng.normal(size=length)
    elif preset=="Peak-Normalized Audio":
        y=0.92*np.sin(2*np.pi*173*t)+0.12*np.sin(2*np.pi*997*t)
        y=y/max(np.max(np.abs(y)),1e-12)*(32767/32768)
    else: raise ValueError(f"Unknown preset: {preset}")
    y=np.clip(np.rint(y*32767),-32768,32767).astype(np.int16)
    if channels==2:
        right=np.roll(y,17)
        return np.column_stack([y,right]).astype(np.int16)
    return y
