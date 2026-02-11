"""Debug script to test attention implementation fix"""

import os

os.environ["TRANSFORMERS_ATTN_IMPLEMENTATION"] = "eager"

import torch

print("[DEBUG] Loading Chatterbox Multilingual TTS model...")
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEBUG] Using device: {device}")

model = ChatterboxMultilingualTTS.from_pretrained(device=device)

print(f"[DEBUG] Model loaded")

# Force eager attention on the loaded transformer model
if hasattr(model, "t3") and hasattr(model.t3, "tfmr"):
    tfmr = model.t3.tfmr
    print(f"[DEBUG] Transformer type: {type(tfmr)}")
    if hasattr(tfmr, "config"):
        config = tfmr.config
        print(
            f"[DEBUG] BEFORE - Transformer config _attn_implementation: {getattr(config, '_attn_implementation', 'not set')}"
        )

        # Force to eager
        config._attn_implementation = "eager"

        print(
            f"[DEBUG] AFTER - Transformer config _attn_implementation: {getattr(config, '_attn_implementation', 'not set')}"
        )

    # Check the actual layers
    if hasattr(tfmr, "layers") or hasattr(tfmr, "h"):
        layers = getattr(tfmr, "layers", None) or getattr(tfmr, "h", None)
        if layers and len(layers) > 0:
            first_layer = layers[0]
            print(f"[DEBUG] First layer type: {type(first_layer)}")
            if hasattr(first_layer, "attn") or hasattr(first_layer, "self_attn"):
                attn = getattr(first_layer, "attn", None) or getattr(
                    first_layer, "self_attn", None
                )
                print(f"[DEBUG] Attention module type: {type(attn)}")

# Try generating
print("\n[DEBUG] Testing generation...")
try:
    text = "Hello, this is a test."
    wav = model.generate(text, language_id="en", temperature=0.7)
    print(f"[DEBUG] ✓ Generation successful! Audio shape: {wav.shape}")
except Exception as e:
    print(f"[DEBUG] ✗ Generation failed with error: {e}")
    import traceback

    traceback.print_exc()
