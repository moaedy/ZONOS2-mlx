# ZONOS2-mlx

**ZONOS2 text-to-speech running natively on Apple Silicon via [MLX](https://github.com/ml-explore/mlx) - real-time generation and voice cloning, no CUDA required.**

[ZONOS2](https://huggingface.co/Zyphra/ZONOS2) is Zyphra's open-weight, ~8B sparse-MoE text-to-speech model. Its official inference stack is **Linux + NVIDIA only** - it depends on `flashinfer`, `sgl_kernel`, `cutlass`, custom CUDA kernels and a Mini-SGLang serving engine, none of which exist on a Mac.

This project reimplements the entire model in **pure MLX**, Apple's array framework built for Apple Silicon, so the **real `Zyphra/ZONOS2` checkpoint runs natively on an M-series Mac at real-time speed** - including voice cloning.

> A PyTorch/Metal (MPS) sibling port lives at **[ZONOS2-mps](https://github.com/moaedy/ZONOS2-mps)**. This MLX backend is ~3x faster.

---

## Performance (Apple M5 Max, 128 GB, macOS, MLX 0.31, bf16)

| Metric | MLX (this repo) | PyTorch / MPS | naive fp32 |
|---|---|---|---|
| **Decode speed** | **~85-93 frames/s** | ~28 frames/s | ~9 frames/s |
| **Real-time?** | **Yes** (44.1 kHz audio = 86 frames/s) | ~0.33x | ~0.10x |
| Checkpoint load | ~6.5 s | ~3 s | ~7 s |
| Resident memory | ~15 GB (bf16, unified) | ~15 GB | ~30 GB |

A 5-second clip is generated in about 5 seconds. The audio frame rate of the DAC codec is 44100 / 512 = **86.1 frames per second**, so ~85-93 frames/s of generation is **real time**.

With **4-bit quantization** (`--quantize 4`) it gets faster *and* smaller, with no audible quality loss:

| 4-bit (`--quantize 4`) | value |
|---|---|
| Decode speed | **~100 frames/s** (above real time) |
| Resident memory | **~6 GB** (vs ~15 GB bf16) |
| Quality vs bf16 | prefill logit cosine **0.9996** |

This makes the ~8B model comfortable on **16 GB** Macs. See [Quantization](#quantization) below.

**Correctness is verified, not assumed.** The MLX forward pass is checked against the PyTorch reference port on identical inputs:

```
prefill argmax per codebook : identical (all 9 codebooks)
per-codebook logit cosine   : 1.0000
max abs logit difference    : 0.31   (bf16 backend noise; logits soft-capped to +-15)
```

Voice cloning is verified by speaker-embedding cosine - a clip cloned from a female reference scores **0.974** against that reference vs **0.921** against a male reference (closest to its own source).

---

## How it works

ZONOS2 generates audio autoregressively: at each step it emits one frame of **9 DAC codebook tokens**, which a neural audio codec turns into 44.1 kHz waveform. The language model is a 28-layer transformer with a 16-expert mixture-of-experts. This repo runs that whole loop in MLX.

```
text ──> UTF-8 byte tokens + quality/silence conditioning
              │
              ▼
   ┌──────────────────────────────────────────────┐
   │  ZONOS2 LM  (pure MLX, runs on the GPU)        │
   │  28 x { GQA attention + Sonic-MoE / dense FFN }│  ◄── optional speaker
   │  emits 9 codebook logits per step              │      embedding injected
   └──────────────────────────────────────────────┘      at the prompt head
              │  autoregressive decode (delay pattern)
              ▼
   9-codebook frames ──> DAC vocoder (44.1 kHz) ──> waveform
```

### What runs in MLX
The full language model - every matmul on the hot path:

- **GQA attention** (16 query / 4 KV heads) with QK-RMSNorm, a learned per-head temperature, headwise sigmoid gating, and interleaved (GPT-J) RoPE. Uses MLX's native `mx.fast.scaled_dot_product_attention` with grouped-query support and a `"causal"` mask for prefill.
- **Sonic mixture-of-experts** (16 experts, top-1 routing; one layer routes top-2) with an EDA router. The expert dispatch uses MLX's fused grouped-matmul, **`mx.gather_mm`**, which selects each token's expert weights by index *without materializing* them - the idiomatic, fast MLX MoE primitive (the same one `mlx-lm` uses for Qwen3-MoE/Mixtral). This is what keeps top-1 decode cheap: only the one selected expert per token is computed.
- **Preallocated KV cache** with in-place slice writes (no per-step `concatenate`), so each decode step stays launch-light.
- RMSNorm and the router softmax run in float32 for stability; everything else is bf16.

### What stays in PyTorch
Two things that run **once**, off the hot loop, so the framework doesn't matter:

- the **DAC vocoder** ([descript-audio-codec](https://github.com/descriptinc/descript-audio-codec), 44.1 kHz), which turns the generated codes into audio, and
- the **speaker encoder** for voice cloning (an ECAPA-TDNN distributed as `marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B`). Its 2048-d embedding is projected by the checkpoint's own LDA + speaker projection and injected at the prompt head; embeddings are cached next to the audio as `*.zonos2spk.npy`.

### Why MLX is faster than PyTorch-on-Metal here
Autoregressive decode is **launch-bound and memory-bound**: each token is dozens of tiny ops and re-reads the weights. PyTorch on Apple Silicon routes an NVIDIA-first framework through Metal eagerly, paying dispatch overhead on every op. MLX is native to Apple Silicon - unified memory (no host/device copies), lower per-op dispatch cost, and op fusion - which is exactly the regime AR decode lives in. The fused `mx.gather_mm` MoE and the static-shape KV cache remove the remaining hot spots.

---

## Install

Requires an Apple Silicon Mac (M1 or newer) and Python 3.10+.

```bash
git clone https://github.com/moaedy/ZONOS2-mlx.git
cd ZONOS2-mlx
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The ~15 GB `Zyphra/ZONOS2` checkpoint is downloaded automatically from Hugging Face on first run.

## Demo

The easiest way to try it - type text, pick a voice from the dropdown, press **Generate**, and it plays:

```bash
python app.py                  # web UI (needs `pip install gradio`)
python app.py --quantize 4     # same, in ~6 GB

python demo.py                 # terminal version, no extra deps (plays via afplay)
```

`app.py` opens a local Gradio page with a text box, a **voice dropdown** populated from `default_voices/`, a **🎙️ record/upload box** (record or drop in a clip to clone *your own* voice on the fly - it overrides the dropdown), temperature/seed controls, and two buttons:

- **Generate** - returns the whole clip.
- **Stream ▶** - plays the audio **as it is generated**. Because decode runs at ~real time, sound starts in about **1.4 s** and then keeps pace with playback. (Implemented by decoding the codebook frames incrementally as they come out of the autoregressive loop.)

The model loads once and stays resident, so every generation after the first is fast. `demo.py` is a zero-dependency terminal loop (`:voice` to switch voice, `:quit` to exit). Drop any `.wav`/`.mp3` into `default_voices/` and it appears in the dropdown as a clonable voice.

## Usage (CLI)

```bash
# Text-to-speech (default voice)
python mlx_zonos2.py --text "Hello from Zonos 2, running natively on Apple Silicon." --out out.wav

# Voice cloning from a short reference clip
python mlx_zonos2.py \
  --text "This was cloned from a few seconds of reference audio." \
  --voice default_voices/AmericanFemale.mp3 --out clone.wav
```

Useful flags:

| Flag | Default | Notes |
|---|---|---|
| `--text` | demo line | Text to synthesize |
| `--voice` | none | Reference audio for cloning (`.wav/.mp3/.flac/...`) |
| `--dtype` | `bf16` | `bf16` is fastest **and** highest quality on Apple Silicon; `fp16`/`fp32` also work |
| `--seed` | `42` | Reproducibility |
| `--temperature` | `1.15` | Sampling temperature |
| `--max-tokens` | `900` | ~10.5 s of audio cap |
| `--quantize` | `none` | `4` or `8` to quantize the weights (see below) |
| `--speaker-device` | `cpu` | Device for the one-shot speaker encoder |

Example outputs are in [`samples/`](samples/).

## Quantization

```bash
python mlx_zonos2.py --text "..." --quantize 4 --out out.wav   # ~6 GB, ~100 frames/s
python mlx_zonos2.py --text "..." --quantize 8 --out out.wav   # ~7.5 GB, lossless
```

MLX makes weight quantization a one-liner (`mx.quantize`) with fused low-bit kernels (`mx.quantized_matmul` for the linears, **`mx.gather_qmm`** for the MoE experts - the quantized sibling of `gather_mm`). But naive 4-bit-everything collapses this model, so `--quantize 4` uses a **tuned mixed scheme**, arrived at empirically by measuring prefill-logit cosine against the bf16 reference:

| Weights | Precision | Why |
|---|---|---|
| MoE experts (the bulk, ~7 B params) | **4-bit, group 32** | 16x-redundant; tolerates 4-bit *if* groups are small (group-64 collapses to cosine 0.84, group-32 holds at **0.9997**) |
| Attention (`wq/wkv/wo`) | 8-bit | QK-norm + learned temperature make attention precision-sensitive |
| Output head + early dense FFNs | 8-bit | codebook logits are sensitive; dense layers are early, so their error compounds |
| Router, norms, gater, embeddings | bf16 | tiny, and routing accuracy is critical |

Result: **~6 GB resident, ~100 frames/s, prefill logit cosine 0.9996 vs bf16** - no audible quality loss, and it fits a 16 GB Mac. `--quantize 8` is bit-for-bit lossless (cosine 1.000) at ~7.5 GB.

## Long text

A single pass is bounded by the model's context window: the text bytes **and** the generated audio share the same 6144-frame (~71 s) budget, so one utterance tops out around ~500 characters / ~30-60 s (this matches the reference, which caps `max_tokens` at `max_seq_len`). Text longer than that is **automatically split into sentence-grouped segments** and stitched (with a short silence between), so arbitrarily long input plays start-to-finish - in both the full and streaming paths. Short text stays a single pass for the most natural prosody.

## Limitations
- **No text normalization yet.** The official model uses NeMo TN to verbalize numbers/dates; this port sends raw text bytes, so spell out numbers (`"twenty twenty six"`, not `"2026"`) for best results.
- Single-sequence generation (no request batching).
- Segment boundaries in very long text have a small silence seam (the model has no cross-segment prosody).
- The speaker encoder loads a ~1.7 B model the first time you clone a new voice (then caches the embedding).

## Roadmap
- ~~4-bit quantization~~ - **done** (see [Quantization](#quantization)): ~6 GB, ~100 frames/s, near-lossless.
- NeMo-equivalent text normalization.
- Streaming output and request batching.

## Credits & license
- Model weights and architecture: **[Zyphra/ZONOS2](https://huggingface.co/Zyphra/ZONOS2)** ([repo](https://github.com/Zyphra/ZONOS2)). This project loads the official checkpoint and reimplements inference; it does not redistribute the weights.
- Vocoder: [descript-audio-codec](https://github.com/descriptinc/descript-audio-codec). Speaker encoder: [`marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B`](https://huggingface.co/marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B). Framework: [MLX](https://github.com/ml-explore/mlx).
- This port: MIT licensed (see [LICENSE](LICENSE)).
