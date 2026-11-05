"""Audio Steganography — Streamlit Application."""

from __future__ import annotations

import os
import sys
import subprocess
import html
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from core.audio_io import read_wav, write_wav, synthetic_audio
from core.lwt import roundtrip_error
from core.stego_engine import (
    analyze_coefficients,
    embed,
    extract,
    tamper_samples,
)
from core.metrics import (
    mse,
    psnr,
    snr,
    segmental_snr,
    histogram_error_ratio,
    moments_dr,
)
from core.crypto import validate_password


# ---------------------------------------------------------------------
# Launch support
# ---------------------------------------------------------------------

if (
    __name__ == "__main__"
    and os.environ.get("STREAMLIT_RUN_CONTEXT") != "1"
):
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        in_streamlit = get_script_run_ctx() is not None
    except Exception:
        in_streamlit = False

    if not in_streamlit:
        env = os.environ.copy()
        env["STREAMLIT_RUN_CONTEXT"] = "1"

        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(Path(__file__).resolve()),
            ],
            env=env,
        )
        raise SystemExit


st.set_page_config(
    page_title="Audio Steganography",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------
# CSS styling
# ---------------------------------------------------------------------

st.markdown(
    """
<style>
:root {
    color-scheme: light;
    --ink: #172033;
    --muted: #5c6678;
    --line: #d9e1eb;
    --panel: #ffffff;
    --canvas: #f3f6fa;
    --accent: #244b76;
    --accent-hover: #193957;
}

/* Main page */
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: var(--canvas) !important;
    color: var(--ink) !important;
}

.stApp,
.stApp p,
.stApp label,
.stApp span,
.stApp li,
.stApp small,
.stMarkdown,
.stMarkdown p,
.stMarkdown li,
.stMarkdown label,
[data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"] p,
[data-testid="stMarkdownContainer"] {
    color: var(--ink);
}

/* Header spacing: prevent title from overlapping the top toolbar */
.block-container {
    max-width: 1440px;
    padding-top: 3.25rem !important;
    padding-bottom: 3rem;
}

header[data-testid="stHeader"] {
    background: rgba(243, 246, 250, 0.96) !important;
    height: 3rem !important;
    z-index: 1000;
}

[data-testid="stAppViewContainer"] > .main {
    padding-top: 0.5rem !important;
}

[data-testid="stMainBlockContainer"] {
    padding-top: 1.25rem !important;
}

.brand {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 0 0 0.65rem 0;
    padding: 0.5rem 0 0.65rem;
    position: relative;
    z-index: 1;
}

.brand-title {
    font-size: 1.65rem;
    font-weight: 700;
    letter-spacing: -0.035em;
    color: #172033 !important;
    line-height: 1.4;
    display: block;
    margin: 0 0 0.2rem 0;
}

.brand-subtitle {
    color: #5c6678 !important;
    font-size: 0.9rem;
    margin-top: 4px;
    line-height: 1.5;
}

/* Sections and panels */
.section-title {
    font-size: 1.05rem;
    font-weight: 650;
    color: #172033 !important;
    margin: 0.2rem 0 0.65rem 0;
}

.panel {
    background: #ffffff;
    border: 1px solid #d9e1eb;
    border-radius: 12px;
    padding: 18px 20px;
    margin: 8px 0 16px 0;
}

.eyebrow {
    color: #5c6678 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.72rem;
    font-weight: 700;
}

.small-muted,
.stCaption,
[data-testid="stCaptionContainer"] {
    color: #5c6678 !important;
    font-size: 0.86rem;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #d9e1eb;
    padding: 14px 16px;
    border-radius: 10px;
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] p {
    color: #5c6678 !important;
}

div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] div {
    color: #172033 !important;
    font-size: 1.45rem;
}

/* Inputs */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input {
    background: #ffffff !important;
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
    border: 1px solid #c8d2df !important;
    border-radius: 8px !important;
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #7a8494 !important;
    -webkit-text-fill-color: #7a8494 !important;
}

/* Select boxes */
div[data-baseweb="select"] > div {
    background: #ffffff !important;
    color: #172033 !important;
    border-color: #c8d2df !important;
}

div[data-baseweb="select"] span,
div[data-baseweb="select"] input {
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
}

ul[role="listbox"],
li[role="option"],
div[data-baseweb="popover"] {
    background: #ffffff !important;
    color: #172033 !important;
}

li[role="option"] * {
    color: #172033 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #ffffff !important;
    color: #172033 !important;
    border-radius: 10px !important;
}

[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
    background: #f8fafc !important;
    border: 1px dashed #94a3b8 !important;
    border-radius: 10px !important;
}

[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] * {
    color: #172033 !important;
}

[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] svg {
    color: #244b76 !important;
    opacity: 1 !important;
}

[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] svg path {
    stroke: #244b76 !important;
    opacity: 1 !important;
}

[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p {
    color: #172033 !important;
}

/* Default buttons */
.stButton > button,
.stDownloadButton > button,
button[data-testid="stBaseButton-secondary"],
button[data-testid="stBaseButton-tertiary"] {
    border-radius: 8px !important;
    font-weight: 600 !important;
    min-height: 2.65rem;
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
    background: #ffffff !important;
    border: 1px solid #c8d2df !important;
}

.stButton > button:not([kind="primary"]):hover,
.stDownloadButton > button:not([kind="primary"]):hover {
    border-color: #244b76 !important;
    color: #193957 !important;
    -webkit-text-fill-color: #193957 !important;
}

/* Primary buttons: white text on dark blue, including nested spans/icons */
.stButton > button[kind="primary"],
.stButton > button[kind="primary"] *,
.stDownloadButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] *,
button[kind="primary"],
button[kind="primary"] *,
button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-primary"] *,
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-primary"] * {
    background-color: #244b76 !important;
    border-color: #244b76 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

.stButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover,
button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
    background-color: #193957 !important;
    border-color: #193957 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

.stButton > button[kind="primary"]:focus,
.stDownloadButton > button[kind="primary"]:focus,
button[data-testid="stBaseButton-primary"]:focus {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    outline-color: #244b76 !important;
}

/* Disabled buttons */
.stButton > button:disabled,
.stDownloadButton > button:disabled,
button[data-testid="stBaseButton-primary"]:disabled {
    color: #7b8492 !important;
    -webkit-text-fill-color: #7b8492 !important;
    background: #e8edf3 !important;
    border-color: #d9e1eb !important;
}

/* Back button: keep it blue with white text */
.st-key-back_to_workspace button,
.st-key-back_to_workspace button * {
    background-color: #244b76 !important;
    border-color: #244b76 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

.st-key-back_to_workspace button:hover,
.st-key-back_to_workspace button:hover * {
    background-color: #193957 !important;
    border-color: #193957 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* Upload button: white text on blue */
[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] button * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    background: #244b76 !important;
    border-color: #244b76 !important;
}

[data-testid="stFileUploader"] button:hover,
[data-testid="stFileUploader"] button:hover * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    background: #193957 !important;
    border-color: #193957 !important;
}

/* Tabs and radio controls */
div[data-baseweb="tab-list"] {
    gap: 8px;
}

div[data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    color: #172033 !important;
}

div[data-baseweb="tab"] p {
    color: #172033 !important;
}

div[role="radiogroup"] label,
div[role="radiogroup"] label p {
    color: #172033 !important;
}

/* Expanders and tables */
div[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #d9e1eb !important;
    border-radius: 10px !important;
}

div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary p {
    color: #172033 !important;
}

div[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background: #ffffff !important;
    color: #172033 !important;
}

div[data-testid="stAlert"] p {
    color: #172033 !important;
}

hr {
    border-color: #d9e1eb !important;
    margin-top: 1rem !important;
    margin-bottom: 1.25rem !important;
}

/* Recovered text */
.recovered-text {
    background: #ffffff !important;
    color: #172033 !important;
    border: 1px solid #c8d2df;
    border-radius: 8px;
    padding: 14px 16px;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.95rem;
    line-height: 1.55;
}

/* Password checklist */
.password-checklist {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 6px 14px;
    margin: 8px 0 12px;
}

.password-check {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 10px;
    border: 1px solid #d9e1eb;
    border-radius: 7px;
    background: #ffffff;
    color: #172033 !important;
    line-height: 1.35;
}

.password-check .mark {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex: 0 0 1.2rem;
    width: 1.2rem;
    height: 1.2rem;
    font-family: Arial, "Segoe UI Symbol", sans-serif;
    font-size: 1.05rem;
    font-weight: 800;
    line-height: 1;
    text-align: center;
}

.password-pass .mark {
    color: #15803d !important;
}

.password-fail .mark {
    color: #b91c1c !important;
}

footer {
    visibility: hidden;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------

defaults = {
    "stego_bytes": None,
    "cover_bytes": None,
    "cover_samples": None,
    "stego_samples": None,
    "sample_rate": 22050,
    "password": "",
    "show_forensics": False,
    "show_diagnostics": False,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def fmt(value, suffix=""):
    """Format numeric metrics."""
    if value is None:
        return "Not available"

    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return "Not available"

    if isinstance(value, (float, np.floating)) and np.isinf(value):
        return "Infinity"

    return f"{value:.3f}{suffix}"


def panel_heading(title, description=None):
    """Render a section heading and optional description."""
    st.markdown(
        f'<div class="section-title">{html.escape(title)}</div>',
        unsafe_allow_html=True,
    )

    if description:
        st.markdown(
            f'<div class="small-muted">{html.escape(description)}</div>',
            unsafe_allow_html=True,
        )


def back_to_workspace():
    """Return from a utility workspace to the main workspace."""
    if st.button(
        "← Back to Embed / Extract",
        key="back_to_workspace",
        type="primary",
        use_container_width=False,
    ):
        st.session_state.show_forensics = False
        st.session_state.show_diagnostics = False
        st.rerun()


def password_requirements(password, confirmation):
    """Evaluate password requirements independently."""
    return [
        ("At least 12 characters", len(password) >= 12),
        (
            "At least one uppercase letter (A–Z)",
            any(char.isupper() for char in password),
        ),
        (
            "At least one lowercase letter (a–z)",
            any(char.islower() for char in password),
        ),
        (
            "At least one number (0–9)",
            any(char.isdigit() for char in password),
        ),
        (
            "At least one special character",
            any(not char.isalnum() for char in password),
        ),
        (
            "Password confirmation matches",
            bool(confirmation) and password == confirmation,
        ),
    ]


def render_password_checklist(password, confirmation):
    """Render readable green ticks and red crosses."""
    checks = password_requirements(password, confirmation)

    st.markdown("**Password requirements**")

    checks_html = '<div class="password-checklist">'

    for requirement, passed in checks:
        mark = "✓" if passed else "×"
        state_class = "password-pass" if passed else "password-fail"

        checks_html += (
            f'<div class="password-check {state_class}">'
            f'<span class="mark">{mark}</span>'
            f'<span>{html.escape(requirement)}</span>'
            "</div>"
        )

    checks_html += "</div>"

    st.markdown(checks_html, unsafe_allow_html=True)

    return all(passed for _, passed in checks)


# ---------------------------------------------------------------------
# Header and navigation
# ---------------------------------------------------------------------

left, right = st.columns([7, 2], vertical_alignment="center")

with left:
    st.markdown(
        """
        <div class="brand">
          <div>
            <div class="brand-title">Audio Steganography</div>
            <div class="brand-subtitle">
              Adaptive data embedding with authenticated encryption
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    b1, b2 = st.columns(2)

    with b1:
        if st.button("Forensics", use_container_width=True):
            st.session_state.show_forensics = (
                not st.session_state.show_forensics
            )
            st.session_state.show_diagnostics = False
            st.rerun()

    with b2:
        if st.button("Diagnostics", use_container_width=True):
            st.session_state.show_diagnostics = (
                not st.session_state.show_diagnostics
            )
            st.session_state.show_forensics = False
            st.rerun()

st.divider()


# ---------------------------------------------------------------------
# Forensics workspace
# ---------------------------------------------------------------------

if st.session_state.show_forensics:
    back_to_workspace()

    st.markdown("## Security and forensic analysis")
    st.markdown(
        "Run verification tools when needed. These indicators support "
        "testing; they do not prove that audio is undetectable."
    )

    if st.session_state.stego_samples is None:
        st.info(
            "Create a stego audio file in the main workspace before "
            "running these checks."
        )
    else:
        stego = st.session_state.stego_samples
        cover = st.session_state.cover_samples
        password = st.session_state.password

        c1, c2, c3 = st.columns(3)

        with c1:
            if st.button(
                "Verify authorized extraction",
                use_container_width=True,
            ):
                try:
                    result, _ = extract(stego, password)
                    st.success(
                        "Authenticated recovery succeeded. Payload size: "
                        f"{len(result.payload):,} bytes."
                    )
                except Exception as exc:
                    st.error(f"Verification failed: {exc}")

        with c2:
            if st.button(
                "Test incorrect password",
                use_container_width=True,
            ):
                try:
                    extract(stego, password + "-invalid")
                    st.error("Unexpected result: extraction was accepted.")
                except Exception:
                    st.success("Incorrect password rejected.")

        with c3:
            if st.button(
                "Simulate sample modification",
                use_container_width=True,
            ):
                try:
                    extract(tamper_samples(stego, 0), password)
                    st.warning(
                        "Extraction still succeeded. The sample change "
                        "may not have altered the embedded packet bits."
                    )
                except Exception as exc:
                    st.success(f"Modified audio rejected: {exc}")

        if cover is not None:
            st.divider()

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Measured WAV size difference",
                f"{len(st.session_state.stego_bytes) - len(st.session_state.cover_bytes):,} bytes",
            )

            c2.metric(
                "Histogram error ratio",
                fmt(histogram_error_ratio(cover, stego)),
            )

            c3.metric("Sample MSE", fmt(mse(cover, stego)))

            st.markdown("#### Statistical moment difference")

            st.dataframe(
                pd.DataFrame([moments_dr(cover, stego)]),
                use_container_width=True,
                hide_index=True,
            )

    st.stop()


# ---------------------------------------------------------------------
# Diagnostics workspace
# ---------------------------------------------------------------------

if st.session_state.show_diagnostics:
    back_to_workspace()

    st.markdown("## System diagnostics")
    st.markdown(
        "Run a compact functional smoke test across the built-in carrier presets."
    )

    if st.button("Run diagnostic suite", type="primary"):
        rows = []

        diagnostic_password = "Diagnostic-Password-2026!"
        test_payload = b"Audio steganography test payload"

        presets = [
            "Speech Formant",
            "Classical Harmonic",
            "High-Energy Pop",
            "Percussive Jazz",
            "Peak-Normalized Audio",
        ]

        for preset in presets:
            try:
                samples = synthetic_audio(preset, seconds=3)
                info = analyze_coefficients(samples)

                stego, _ = embed(
                    samples,
                    diagnostic_password,
                    test_payload,
                )

                result, _ = extract(stego, diagnostic_password)

                rows.append(
                    {
                        "Carrier preset": preset,
                        "Capacity (bytes)": int(
                            info["depths"].sum() // 8
                        ),
                        "MSE": mse(samples, stego),
                        "PSNR (dB)": psnr(samples, stego),
                        "Recovery": (
                            "PASS"
                            if result.payload == test_payload
                            else "FAIL"
                        ),
                        "PCM range": (
                            "PASS"
                            if stego.min() >= -32768
                            and stego.max() <= 32767
                            else "FAIL"
                        ),
                    }
                )

            except Exception as exc:
                rows.append(
                    {
                        "Carrier preset": preset,
                        "Capacity (bytes)": 0,
                        "MSE": None,
                        "PSNR (dB)": None,
                        "Recovery": f"FAIL: {exc}",
                        "PCM range": "Not checked",
                    }
                )

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

    st.stop()


# ---------------------------------------------------------------------
# Main workspace selector
# ---------------------------------------------------------------------

workspace = st.radio(
    "Workspace",
    ["Embed payload", "Extract payload"],
    horizontal=True,
    label_visibility="collapsed",
)


# ---------------------------------------------------------------------
# EMBED PAYLOAD WORKSPACE
# ---------------------------------------------------------------------

if workspace == "Embed payload":
    st.markdown("## Embed a payload")

    st.markdown(
        "Select a carrier WAV, review its available capacity, then protect "
        "and embed a text message or file."
    )

    with st.container(border=True):
        panel_heading(
            "1. Carrier audio",
            "Use an uncompressed 16-bit PCM WAV or generate a sample carrier.",
        )

        source_col, settings_col = st.columns([1.35, 1])

        with source_col:
            mode = st.radio(
                "Carrier source",
                ["Upload WAV", "Synthetic preset"],
                horizontal=True,
                key="carrier_mode",
            )

            if mode == "Upload WAV":
                uploaded = st.file_uploader(
                    "Choose 16-bit PCM WAV",
                    type=["wav"],
                    key="cover_upload",
                )

                if uploaded:
                    try:
                        new_bytes = uploaded.getvalue()
                        cover, rate = read_wav(new_bytes)

                        if (
                            st.session_state.get("loaded_cover_name")
                            != uploaded.name
                            or st.session_state.get("loaded_cover_bytes")
                            != new_bytes
                        ):
                            st.session_state.cover_samples = cover
                            st.session_state.sample_rate = rate
                            st.session_state.cover_bytes = new_bytes
                            st.session_state.carrier_label = uploaded.name
                            st.session_state.loaded_cover_name = uploaded.name
                            st.session_state.loaded_cover_bytes = new_bytes

                            st.session_state.stego_samples = None
                            st.session_state.stego_bytes = None
                            st.session_state.pop("embed_meta", None)

                    except Exception as exc:
                        st.error(f"Could not read carrier WAV: {exc}")

            else:
                p1, p2 = st.columns([1.3, 1])

                with p1:
                    preset = st.selectbox(
                        "Carrier profile",
                        [
                            "Speech Formant",
                            "Classical Harmonic",
                            "High-Energy Pop",
                            "Percussive Jazz",
                            "Peak-Normalized Audio",
                        ],
                    )

                with p2:
                    channels = st.selectbox(
                        "Channel layout",
                        [1, 2],
                        format_func=lambda value: (
                            "Mono" if value == 1 else "Stereo"
                        ),
                    )

                seconds = st.slider(
                    "Duration",
                    2,
                    30,
                    8,
                    format="%d seconds",
                )

                if st.button("Generate carrier", type="secondary"):
                    try:
                        cover = synthetic_audio(
                            preset,
                            seconds,
                            22050,
                            channels,
                        )

                        cover_bytes = write_wav(cover, 22050)

                        st.session_state.cover_samples = cover
                        st.session_state.sample_rate = 22050
                        st.session_state.cover_bytes = cover_bytes
                        st.session_state.carrier_label = (
                            f"{preset} · {seconds}s"
                        )

                        st.session_state.stego_samples = None
                        st.session_state.stego_bytes = None
                        st.session_state.pop("embed_meta", None)

                        st.rerun()

                    except Exception as exc:
                        st.error(f"Could not generate carrier: {exc}")

        with settings_col:
            if st.session_state.cover_samples is not None:
                cover = st.session_state.cover_samples
                rate = st.session_state.sample_rate
                info = analyze_coefficients(cover)

                capacity_bits = int(info["depths"].sum())
                capacity_bytes = capacity_bits // 8

                st.markdown(
                    '<div class="eyebrow">Carrier summary</div>',
                    unsafe_allow_html=True,
                )

                st.metric(
                    "Usable payload capacity",
                    f"{capacity_bytes:,} bytes",
                )

                # Calculate duration from the currently loaded audio,
                # rather than trusting the selected slider value.
                frame_count = cover.shape[0]
                actual_duration = (
                    frame_count / rate if rate > 0 else 0.0
                )

                st.caption(
                    f"{rate:,} Hz · "
                    f"{'Mono' if cover.ndim == 1 else 'Stereo'} · "
                    f"{frame_count:,} frames · "
                    f"{actual_duration:.2f} seconds"
                )

                if st.session_state.get("carrier_label"):
                    st.caption(
                        f"Carrier: {st.session_state.carrier_label}"
                    )

                st.caption(
                    "Baseline transform reconstruction error: "
                    f"{roundtrip_error(cover)}"
                )

                st.caption(
                    f"Available coefficient depths: "
                    f"{info['counts'][1]:,} at 1 bit, "
                    f"{info['counts'][2]:,} at 2 bits, "
                    f"{info['counts'][3]:,} at 3 bits."
                )

            else:
                st.info(
                    "Choose or generate a carrier to view its capacity "
                    "and format details."
                )

    # Payload and protection
    if st.session_state.cover_samples is not None:
        cover = st.session_state.cover_samples
        rate = st.session_state.sample_rate

        with st.container(border=True):
            panel_heading(
                "2. Payload and protection",
                "Your payload is encrypted before being embedded into the carrier.",
            )

            kind = st.radio(
                "Payload type",
                ["Text message", "Binary file"],
                horizontal=True,
                key="payload_kind",
            )

            if kind == "Text message":
                text_payload = st.text_area(
                    "Message to conceal",
                    placeholder="Enter the message you want to embed.",
                    height=120,
                    key="message_to_embed",
                )

                payload = text_payload.encode("utf-8")
                filename = ""

            else:
                upload_payload = st.file_uploader(
                    "Choose a file to conceal",
                    key="payload_file",
                )

                payload = (
                    upload_payload.getvalue()
                    if upload_payload
                    else None
                )

                filename = (
                    upload_payload.name
                    if upload_payload
                    else ""
                )

                if upload_payload:
                    st.caption(
                        f"Selected: {filename} · {len(payload):,} bytes"
                    )

            pw1 = st.text_input(
                "Encryption password",
                type="password",
                key="embed_password",
                help=(
                    "Use at least 12 characters, including uppercase, "
                    "lowercase, a number, and a special character."
                ),
            )

            pw2 = st.text_input(
                "Confirm password",
                type="password",
                key="confirm_password",
            )

            # Each password rule has its own visible pass/fail state.
            password_ok = render_password_checklist(pw1, pw2)

            try:
                validate_password(pw1)
            except ValueError:
                password_ok = False

            payload_ok = payload is not None and len(payload) > 0

            if payload is not None:
                packet_estimate = (
                    43
                    + len(filename.encode("utf-8"))
                    + len(payload)
                    + 16
                )

                st.caption(
                    f"Estimated packet size: {packet_estimate:,} bytes, "
                    "before capacity verification."
                )

            if st.button(
                "Encrypt and embed",
                type="primary",
                use_container_width=True,
                disabled=not password_ok or not payload_ok,
                key="encrypt_embed_button",
            ):
                try:
                    stego, meta = embed(
                        cover,
                        pw1,
                        payload,
                        0 if kind == "Text message" else 1,
                        filename,
                    )

                    stego_bytes = write_wav(stego, rate)

                    st.session_state.stego_samples = stego
                    st.session_state.stego_bytes = stego_bytes
                    st.session_state.password = pw1
                    st.session_state.embed_meta = meta

                    st.success(
                        "Payload embedded successfully. Encrypted packet: "
                        f"{meta['packet_bytes']:,} bytes."
                    )

                except Exception as exc:
                    st.error(f"Embedding failed: {exc}")

    # Download and embedded audio analysis
    if (
        st.session_state.stego_bytes is not None
        and st.session_state.stego_samples is not None
        and st.session_state.cover_samples is not None
    ):
        st.markdown("## Embedded audio")

        st.success(
            "Stego audio generated successfully. Your file is ready to download."
        )

        st.download_button(
            label="⬇️ Download Stego WAV",
            data=st.session_state.stego_bytes,
            file_name="stego_audio.wav",
            mime="audio/wav",
            type="primary",
            key="download_stego_wav",
            use_container_width=True,
        )

        cover = st.session_state.cover_samples
        stego = st.session_state.stego_samples
        meta = st.session_state.get("embed_meta", {})

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("PSNR", fmt(psnr(cover, stego), " dB"))
        m2.metric("SNR", fmt(snr(cover, stego), " dB"))
        m3.metric(
            "Segmental SNR",
            fmt(segmental_snr(cover, stego), " dB"),
        )
        m4.metric("Mean squared error", fmt(mse(cover, stego)))

        original_col, stego_col = st.columns(2)

        with original_col:
            panel_heading("Original carrier")
            st.audio(
                st.session_state.cover_bytes,
                format="audio/wav",
            )

        with stego_col:
            panel_heading("Stego audio")
            st.audio(
                st.session_state.stego_bytes,
                format="audio/wav",
            )

        with st.expander("View signal comparison"):
            a = np.asarray(cover).reshape(-1)
            b = np.asarray(stego).reshape(-1)
            n = min(3000, len(a), len(b))

            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(a[:n], label="Original", linewidth=1)
            ax.plot(
                b[:n],
                label="Stego",
                linewidth=1,
                alpha=0.8,
            )
            ax.set_title("Waveform comparison")
            ax.set_xlabel("Sample index")
            ax.set_ylabel("Amplitude")
            ax.legend()

            st.pyplot(fig)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(10, 2.5))
            ax.plot((b - a)[:n], linewidth=0.8)
            ax.set_title("Sample-level modification")
            ax.set_xlabel("Sample index")
            ax.set_ylabel("Difference")

            st.pyplot(fig)
            plt.close(fig)

        with st.expander("View embedding analysis"):
            info = analyze_coefficients(cover)

            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "1-bit candidates": info["counts"][1],
                            "2-bit candidates": info["counts"][2],
                            "3-bit candidates": info["counts"][3],
                            "PCM-boundary rejected": info["pcm_rejected"],
                            "Quiet-region rejected": info["silent_rejected"],
                            "Bits embedded": meta.get("embedded_bits", 0),
                        }
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )


# ---------------------------------------------------------------------
# EXTRACT PAYLOAD WORKSPACE
# ---------------------------------------------------------------------

else:
    st.markdown("## Extract a payload")

    st.markdown(
        "Load a stego WAV and authenticate the hidden payload with "
        "the original password."
    )

    with st.container(border=True):
        source = st.radio(
            "Stego audio source",
            [
                "Use audio from this session",
                "Upload stego WAV",
            ],
            horizontal=True,
            key="extract_source",
        )

        external = (
            st.file_uploader(
                "Choose stego WAV",
                type=["wav"],
                key="extract_upload",
            )
            if source == "Upload stego WAV"
            else None
        )

        password = st.text_input(
            "Decryption password",
            type="password",
            key="extract_password",
        )

        if st.button(
            "Extract and authenticate",
            type="primary",
            use_container_width=True,
            key="extract_button",
        ):
            try:
                if source == "Use audio from this session":
                    if st.session_state.stego_samples is None:
                        raise ValueError(
                            "No stego audio is available in this session."
                        )

                    samples = st.session_state.stego_samples

                else:
                    if external is None:
                        raise ValueError(
                            "Choose a stego WAV file first."
                        )

                    samples, _ = read_wav(external.getvalue())

                result, _ = extract(samples, password)

                st.success(
                    "Authentication succeeded. The AES-GCM tag is valid."
                )

                if result.payload_type == 0:
                    st.markdown("#### Recovered text")

                    recovered_text = result.payload.decode(
                        "utf-8",
                        errors="replace",
                    )

                    st.markdown(
                        '<div class="recovered-text">'
                        + html.escape(recovered_text)
                        + "</div>",
                        unsafe_allow_html=True,
                    )

                else:
                    st.markdown("#### Recovered file")

                    st.write(f"**Filename:** {result.filename}")
                    st.write(
                        f"**Original size:** {len(result.payload):,} bytes"
                    )

                    safe_filename = (
                        Path(result.filename).name or "recovered.bin"
                    )

                    st.download_button(
                        "Download recovered file",
                        data=result.payload,
                        file_name=safe_filename,
                        key="download_recovered_file",
                        mime="application/octet-stream",
                    )

            except Exception as exc:
                st.error(f"Extraction rejected: {exc}")


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------

st.divider()

st.markdown(
    '<div class="small-muted">'
    "Audio Steganography · 16-bit PCM WAV · Integer Haar lifting transform "
    "· AES-256-GCM authenticated encryption"
    "</div>",
    unsafe_allow_html=True,
)