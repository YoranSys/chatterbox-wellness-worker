# Chatterbox TTS Worker for RunPod Serverless
# Text-to-Speech with voice cloning and emotion control
#
# Based on: https://github.com/geronimi73/runpod_chatterbox

# Use RunPod's pre-built PyTorch image (has CUDA, Python 3.11, PyTorch ready)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Chatterbox TTS without deps to avoid PyTorch version conflicts
RUN pip install --no-cache-dir --no-deps chatterbox-tts

# Install chatterbox dependencies (from reference: github.com/geronimi73/runpod_chatterbox)
# These are installed without version pins to work with the base image's PyTorch
RUN pip install --no-cache-dir \
    conformer \
    s3tokenizer \
    librosa \
    resemble-perth \
    huggingface_hub \
    safetensors \
    transformers \
    diffusers \
    einops \
    soundfile \
    scipy \
    omegaconf \
    pyloudnorm

# Install RunPod SDK
RUN pip install --no-cache-dir runpod

# Copy handler
COPY handler.py /app/handler.py

# Pre-download models during build
RUN python - <<'PY'
import torch

_orig_load = torch.load

def _patched_load(*args, **kwargs):
    kwargs.pop("weights_only", None)
    kwargs.setdefault("map_location", "cpu")
    return _orig_load(*args, **kwargs)

torch.load = _patched_load

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("Downloading Chatterbox multilingual model...")
_ = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
print("Chatterbox multilingual model downloaded successfully")
PY

# Start handler
CMD ["python", "-u", "/app/handler.py"]
