
"""Password-based authenticated encryption and packet framing."""
from __future__ import annotations
import struct
from dataclasses import dataclass
from os import urandom
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag

MAGIC = b"ASTG"
HEADER = struct.Struct(">4sB16s12sHII")
HEADER_SIZE = HEADER.size
ITERATIONS = 100_000

class AuthenticationError(ValueError):
    """Raised when packet framing, password, or authenticated data is invalid."""

@dataclass(frozen=True)
class Packet:
    payload_type: int
    filename: str
    payload: bytes
    salt: bytes
    nonce: bytes

def validate_password(password: str) -> None:
    """Validate a new encryption password using the application's configured policy."""
    import re
    if len(password) < 12:
        raise ValueError("Password must contain at least 12 characters.")
    if not re.search(r"[a-z]", password):
        raise ValueError("Password must include at least one lowercase letter.")
    if not re.search(r"[A-Z]", password):
        raise ValueError("Password must include at least one uppercase letter.")
    if not re.search(r"\d", password):
        raise ValueError("Password must include at least one number.")
    if not re.search(r"[^A-Za-z0-9]", password):
        raise ValueError("Password must include at least one special character.")

def derive_key(password: str, salt: bytes) -> bytes:
    if not password:
        raise ValueError("Password must not be empty.")
    if len(salt) != 16:
        raise ValueError("Salt must be 16 bytes.")
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITERATIONS)
    return kdf.derive(password.encode("utf-8"))

def pack_packet(payload: bytes, password: str, payload_type: int, filename: str = "") -> bytes:
    validate_password(password)
    if payload_type not in (0, 1):
        raise ValueError("payload_type must be 0 (text) or 1 (binary file).")
    if payload_type == 0:
        filename = ""
    name = filename.encode("utf-8")
    if len(name) > 65535:
        raise ValueError("UTF-8 filename is too long.")
    if len(payload) > 0xFFFFFFFF:
        raise ValueError("Payload exceeds supported 4 GiB framing limit.")
    salt, nonce = urandom(16), urandom(12)
    key = derive_key(password, salt)
    ciphertext = AESGCM(key).encrypt(nonce, payload, name)
    header = HEADER.pack(MAGIC, payload_type, salt, nonce, len(name), len(ciphertext), len(payload))
    return header + name + ciphertext

def unpack_header(data: bytes):
    if len(data) < HEADER_SIZE:
        raise AuthenticationError("Incomplete header.")
    try:
        magic, ptype, salt, nonce, fname_len, cipher_len, orig_len = HEADER.unpack(data[:HEADER_SIZE])
    except struct.error as exc:
        raise AuthenticationError("Malformed packet header.") from exc
    if magic != MAGIC or ptype not in (0, 1):
        raise AuthenticationError("Incorrect password or no hidden payload found.")
    if ptype == 0 and fname_len != 0:
        raise AuthenticationError("Malformed text packet.")
    if cipher_len < 16:
        raise AuthenticationError("Malformed ciphertext length.")
    return magic, ptype, salt, nonce, fname_len, cipher_len, orig_len

def decrypt_packet(packet: bytes, password: str) -> Packet:
    try:
        magic, ptype, salt, nonce, fname_len, cipher_len, orig_len = unpack_header(packet)
        total = HEADER_SIZE + fname_len + cipher_len
        if len(packet) != total:
            raise AuthenticationError("Packet length does not match its header.")
        name_bytes = packet[HEADER_SIZE:HEADER_SIZE+fname_len]
        ciphertext = packet[HEADER_SIZE+fname_len:]
        filename = name_bytes.decode("utf-8")
        key = derive_key(password, salt)
        payload = AESGCM(key).decrypt(nonce, ciphertext, name_bytes)
        if len(payload) != orig_len:
            raise AuthenticationError("Recovered payload length does not match header.")
        return Packet(ptype, filename, payload, salt, nonce)
    except InvalidTag as exc:
        raise AuthenticationError("Authentication failed: incorrect password or audio/payload tampering detected.") from exc
    except UnicodeDecodeError as exc:
        raise AuthenticationError("Invalid authenticated filename encoding.") from exc
