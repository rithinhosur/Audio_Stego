"""Adaptive embedding and blind extraction for 16-bit PCM WAV samples."""
from __future__ import annotations
import hashlib
import numpy as np
from .lwt import forward_lwt, inverse_lwt, roundtrip_error
from .crypto import HEADER_SIZE, HEADER, AuthenticationError, pack_packet, decrypt_packet, unpack_header

PCM_MIN, PCM_MAX = -32768, 32767

def analyze_coefficients(samples: np.ndarray) -> dict:
    x = np.asarray(samples)
    if x.dtype.kind not in "iu":
        raise ValueError("PCM samples must be integer-valued.")
    if x.size and (int(x.min()) < PCM_MIN or int(x.max()) > PCM_MAX):
        raise ValueError("Input samples are outside signed 16-bit PCM range.")
    s, d, rem = forward_lwt(x)
    base = d & ~np.int64(7)
    dmin, dmax = base, base + 7
    emin = s - (dmin >> 1); omin = emin + dmin
    emax = s - (dmax >> 1); omax = emax + dmax
    safe = ((emin >= PCM_MIN) & (emin <= PCM_MAX) & (omin >= PCM_MIN) & (omin <= PCM_MAX) &
            (emax >= PCM_MIN) & (emax <= PCM_MAX) & (omax >= PCM_MIN) & (omax <= PCM_MAX))
    active = (np.abs(s) >= 64) | (np.abs(base) >= 16)
    depths = np.zeros(d.shape, dtype=np.uint8)
    eligible = safe & active
    magnitude = np.abs(base)
    depths[eligible & (magnitude < 512)] = 1
    depths[eligible & (magnitude >= 512) & (magnitude < 2048)] = 2
    depths[eligible & (magnitude >= 2048)] = 3
    return {"s":s, "d":d, "remainder":rem, "base":base, "safe":safe, "active":active,
            "depths":depths, "pcm_rejected":int(np.count_nonzero(~safe)),
            "silent_rejected":int(np.count_nonzero(safe & ~active)),
            "counts":{k:int(np.count_nonzero(depths == k)) for k in (0,1,2,3)}}

def _ordered(depths: np.ndarray, password: str) -> np.ndarray:
    indices = np.flatnonzero(depths.ravel() > 0)
    seed = int.from_bytes(hashlib.sha256(b"ORDER_SEED_" + password.encode("utf-8")).digest()[:8], "big")
    np.random.default_rng(seed).shuffle(indices)
    return indices

def _bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")

def _from_bits(bits: list[int] | np.ndarray) -> bytes:
    arr = np.asarray(bits, dtype=np.uint8)
    if len(arr) % 8:
        raise ValueError("Bitstream length must be byte-aligned.")
    return np.packbits(arr, bitorder="big").tobytes()

def _write_chunks(original_d: np.ndarray, depths: np.ndarray, order: np.ndarray, bitstream: np.ndarray) -> tuple[np.ndarray, dict]:
    flat = original_d.ravel().copy()
    pos = 0
    used = {1:0, 2:0, 3:0}
    for idx in order:
        if pos >= len(bitstream):
            break
        k = int(depths.ravel()[idx])
        take = min(k, len(bitstream)-pos)
        value_bits = bitstream[pos:pos+take]
        value = 0
        for bit in value_bits:
            value = (value << 1) | int(bit)
        value <<= (k-take)
        flat[idx] = (flat[idx] & ~((1 << k)-1)) | value
        used[k] += 1
        pos += take
    if pos < len(bitstream):
        raise ValueError(f"Payload exceeds capacity: needs {len(bitstream)} bits, embedded {pos}.")
    return flat.reshape(original_d.shape), used

def embed(samples: np.ndarray, password: str, payload: bytes, payload_type: int = 0, filename: str = "") -> tuple[np.ndarray, dict]:
    x = np.asarray(samples)
    info = analyze_coefficients(x)
    packet = pack_packet(payload, password, payload_type, filename)
    bits = _bits(packet)
    order = _ordered(info["depths"], password)
    if len(bits) > int(info["depths"].sum()):
        raise ValueError(f"Payload too large. Packet needs {len(bits)} bits; carrier capacity is {int(info['depths'].sum())} bits.")
    new_d, used = _write_chunks(info["d"], info["depths"], order, bits)
    reconstructed = inverse_lwt(info["s"], new_d, info["remainder"])
    if reconstructed.size and (reconstructed.min() < PCM_MIN or reconstructed.max() > PCM_MAX):
        raise RuntimeError("Internal safety invariant failed: reconstructed PCM is out of range.")
    stego = reconstructed.astype(np.int16)
    return stego, {"packet_bytes":len(packet), "embedded_bits":len(bits), "used_depth":used, "analysis":info}

def extract(samples: np.ndarray, password: str) -> tuple[object, dict]:
    x = np.asarray(samples)
    info = analyze_coefficients(x)
    order = _ordered(info["depths"], password)
    flat = info["d"].ravel()
    depths = info["depths"].ravel()
    collected: list[int] = []
    header = None
    needed_bits = HEADER_SIZE * 8
    total_bits = 0
    for idx in order:
        k = int(depths[idx])
        value = int(flat[idx]) & ((1 << k)-1)
        for shift in range(k-1, -1, -1):
            collected.append((value >> shift) & 1)
        total_bits += k
        if header is None and total_bits >= needed_bits:
            header_bytes = _from_bits(collected[:needed_bits])
            try:
                _, ptype, salt, nonce, fname_len, cipher_len, orig_len = unpack_header(header_bytes)
            except Exception as exc:
                raise AuthenticationError("Incorrect password or no hidden payload found.") from exc
            total_bytes = HEADER_SIZE + fname_len + cipher_len
            needed_bits = total_bytes * 8
            header = (ptype, salt, nonce, fname_len, cipher_len, orig_len)
        if header is not None and len(collected) >= needed_bits:
            packet = _from_bits(collected[:needed_bits])
            result = decrypt_packet(packet, password)
            return result, {"packet_bytes":len(packet), "extracted_bits":needed_bits}
    raise AuthenticationError("Incomplete payload or incorrect password: insufficient readable data.")

def tamper_samples(samples: np.ndarray, sample_index: int = 0) -> np.ndarray:
    """Flip one sample LSB; whether this touches payload bits depends on the selected sample."""
    out = np.asarray(samples, dtype=np.int16).copy()
    if out.size == 0:
        raise ValueError("Cannot tamper with empty audio.")
    flat = out.reshape(-1)
    flat[sample_index % flat.size] = np.int16(int(flat[sample_index % flat.size]) ^ 1)
    return out
