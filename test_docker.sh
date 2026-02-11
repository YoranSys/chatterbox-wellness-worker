#!/bin/bash
# Test the attention fix in Docker

echo "Building Docker image..."
docker build -t tts-test .

echo ""
echo "Running debug script in Docker (CPU mode)..."
docker run --rm tts-test python /app/debug_attention.py
