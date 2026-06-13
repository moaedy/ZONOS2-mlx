#!/usr/bin/env python
"""Terminal demo for ZONOS2-mlx - no extra dependencies.

Loads the model once, then loops: pick a voice, type text, and it generates
and plays the audio (via `afplay` on macOS).

    python demo.py                 # full bf16 (best quality)
    python demo.py --quantize 4    # ~6 GB, fits a 16 GB Mac
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import tempfile
import time

import mlx.core as mx
import soundfile as sf

import mlx_zonos2 as X

VOICE_EXTS = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus")


def discover_voices(folder):
    voices = [("Default (no voice clone)", None)]
    for f in sorted(glob.glob(os.path.join(folder, "*"))):
        if f.lower().endswith(VOICE_EXTS):
            voices.append((os.path.splitext(os.path.basename(f))[0], f))
    return voices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantize", default="none", choices=["none", "4", "8"])
    ap.add_argument("--voices-dir", default="default_voices")
    ap.add_argument("--speaker-device", default="cpu")
    ap.add_argument("--temperature", type=float, default=1.15)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    quantize = None if args.quantize == "none" else int(args.quantize)
    ckpt = args.ckpt or glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--Zyphra--ZONOS2/snapshots/*/model.pth"))[0]

    tts = X.Zonos2MLX(ckpt, dtype=mx.bfloat16, quantize=quantize)
    voices = discover_voices(args.voices_dir)
    cur = 0
    seed = 42

    def show_voices():
        print("\nVoices:")
        for i, (name, _) in enumerate(voices):
            print(f"  {'*' if i == cur else ' '} [{i}] {name}")

    print("\n" + "=" * 60)
    print("  ZONOS2-mlx terminal demo")
    print("  Type text and press Enter to speak it.")
    print("  Commands:  :voice   choose a voice")
    print("             :seed N  set the random seed")
    print("             :quit    exit")
    print("=" * 60)
    show_voices()

    while True:
        try:
            line = input(f"\n[{voices[cur][0]}] text> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not line:
            continue
        if line in (":quit", ":q", ":exit"):
            break
        if line in (":voice", ":v"):
            show_voices()
            sel = input("voice number> ").strip()
            if sel.isdigit() and 0 <= int(sel) < len(voices):
                cur = int(sel)
                print(f"-> {voices[cur][0]}")
            continue
        if line.startswith(":seed"):
            parts = line.split()
            if len(parts) == 2 and parts[1].lstrip("-").isdigit():
                seed = int(parts[1])
                print(f"-> seed {seed}")
            continue

        name, path = voices[cur]
        emb = X.get_speaker_embedding(path, args.speaker_device) if path else None
        frames, eos = tts.generate(line, voice_emb=emb, temperature=args.temperature,
                                   seed=seed, max_tokens=1500, verbose=False)
        audio, sr = X.decode_to_audio(frames, eos, device="mps")
        if audio is None:
            print("(no audio - try different text)")
            continue
        out = os.path.join(tempfile.gettempdir(), f"zonos2_demo_{int(time.time())}.wav")
        sf.write(out, audio, sr)
        print(f"  {len(audio)/sr:.1f}s  ->  {out}")
        try:
            subprocess.run(["afplay", out], check=False)
        except FileNotFoundError:
            print("  (install/enable `afplay`, or open the file above)")


if __name__ == "__main__":
    main()
