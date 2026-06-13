#!/usr/bin/env python
"""Web demo for ZONOS2-mlx: type text, pick a voice, generate and play.

    python app.py                 # full bf16 (best quality; needs ~15 GB)
    python app.py --quantize 4    # ~6 GB, fits a 16 GB Mac

Opens a local Gradio UI. The model loads once and stays resident, so every
generation after the first is fast.

MLX's Metal stream is bound to the thread that creates the model, and Gradio
serves callbacks from a threadpool, so all MLX work is funnelled to one
dedicated worker thread that owns the model.
"""

from __future__ import annotations

import argparse
import glob
import os
import queue
import threading

import gradio as gr
import mlx.core as mx
import numpy as np

import mlx_zonos2 as X

VOICE_EXTS = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus")


def discover_voices(folder: str) -> dict:
    voices = {"Default (no voice clone)": None}
    for f in sorted(glob.glob(os.path.join(folder, "*"))):
        if f.lower().endswith(VOICE_EXTS):
            voices[os.path.splitext(os.path.basename(f))[0]] = f
    return voices


class MLXWorker:
    """Owns the model on a single thread and runs every generation there."""

    def __init__(self, ckpt, dtype, quantize, dac_device="mps"):
        self.dac_device = dac_device
        self._jobs: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self._tts = None
        self._args = (ckpt, dtype, quantize)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait()

    def _run(self):
        ckpt, dtype, quantize = self._args
        self._tts = X.Zonos2MLX(ckpt, dtype=dtype, quantize=quantize)
        self._ready.set()
        while True:
            job = self._jobs.get()
            text, emb, temperature, seed, done = job
            try:
                frames, eos = self._tts.generate(
                    text, voice_emb=emb, temperature=temperature,
                    seed=seed, max_tokens=1500, verbose=False)
                audio, sr = X.decode_to_audio(frames, eos, device=self.dac_device)
                done.result = None if audio is None else (sr, np.asarray(audio, np.float32))
            except Exception as exc:        # surface to the UI thread
                done.error = exc
            done.set()

    def generate(self, text, emb, temperature, seed):
        done = threading.Event()
        done.result = done.error = None
        self._jobs.put((text, emb, temperature, seed, done))
        done.wait()
        if done.error is not None:
            raise done.error
        return done.result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantize", default="none", choices=["none", "4", "8"])
    ap.add_argument("--voices-dir", default="default_voices")
    ap.add_argument("--speaker-device", default="cpu")
    ap.add_argument("--dac-device", default="mps")
    ap.add_argument("--share", action="store_true", help="Create a public Gradio link")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    quantize = None if args.quantize == "none" else int(args.quantize)
    ckpt = args.ckpt or glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--Zyphra--ZONOS2/snapshots/*/model.pth"))[0]

    print("Loading ZONOS2 (this happens once)...", flush=True)
    worker = MLXWorker(ckpt, mx.bfloat16, quantize, dac_device=args.dac_device)
    voices = discover_voices(args.voices_dir)
    print(f"Ready. Voices: {', '.join(voices)}", flush=True)

    def synth(text, voice_name, temperature, seed):
        text = (text or "").strip()
        if not text:
            raise gr.Error("Please enter some text.")
        path = voices.get(voice_name)
        emb = X.get_speaker_embedding(path, args.speaker_device) if path else None
        result = worker.generate(text, emb, float(temperature), int(seed))
        if result is None:
            raise gr.Error("No audio was generated; try different text.")
        return result

    with gr.Blocks(title="ZONOS2-mlx") as demo:
        gr.Markdown(
            "# 🗣️ ZONOS2-mlx\n"
            "ZONOS2 text-to-speech, running natively on Apple Silicon via MLX. "
            "Type text, pick a voice, and press **Generate**."
        )
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label="Text", lines=4,
                    value="Hello! This is Zonos 2, running natively on Apple Silicon with M.L.X.")
            with gr.Column(scale=2):
                voice = gr.Dropdown(
                    choices=list(voices), value=list(voices)[0], label="Voice")
                temperature = gr.Slider(0.5, 1.5, value=1.15, step=0.05, label="Temperature")
                seed = gr.Number(value=42, label="Seed", precision=0)
        btn = gr.Button("Generate", variant="primary")
        out = gr.Audio(label="Output", autoplay=True)
        btn.click(synth, [text, voice, temperature, seed], out)
        text.submit(synth, [text, voice, temperature, seed], out)
        gr.Markdown(
            "_Tip: spell out numbers (\"twenty twenty six\") - text normalization isn't ported yet. "
            "Cloning a brand-new voice loads the speaker encoder once, then caches it._")

    demo.queue(default_concurrency_limit=1).launch(share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
