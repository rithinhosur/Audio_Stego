# Reconstruction-Constrained Adaptive Audio Steganography

Python implementation of password-protected text/file hiding in uncompressed 16-bit PCM WAV audio.

## Install and run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
# Or: python app.py
```

## CLI
```bash
python cli.py inspect cover.wav
python cli.py embed cover.wav stego.wav --password "your password" --text "secret"
python cli.py embed cover.wav stego.wav --password "your password" --file document.pdf
python cli.py extract stego.wav --password "your password" --output-dir recovered
python cli.py verify cover.wav stego.wav
python -m unittest discover -s tests -v
```

## Important engineering notes
- Input is limited to uncompressed 16-bit PCM WAV, mono or stereo.
- The implementation uses the specified 43-byte `>4sB16s12sHII` packet header, PBKDF2-HMAC-SHA256 (100,000 iterations), and AES-256-GCM.
- “0 bytes overhead” means unchanged WAV audio format and frame count; header/container metadata may vary across writers.
- File-size/quality/security outcomes depend on the cover, payload, password, and tampering operation. No universal forensic invisibility or 0.1-second end-to-end guarantee is claimed.
- The password-derived coefficient ordering is for synchronization, not a replacement for cryptographic encryption. Use a strong unique password.
- The requested 0.1-second performance target is hardware- and payload-dependent and should be benchmarked on the deployment machine.


## Password policy
New payloads require a password of at least 12 characters containing uppercase and lowercase letters, a number, and a special character. Extraction accepts the password used when the payload was embedded; do not change it after embedding. This is an application policy, not a claim that every security standard mandates these exact character classes.
