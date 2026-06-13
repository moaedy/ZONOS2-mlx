#!/usr/bin/env python
"""ZONOS2 TTS on Apple Silicon via MLX (Apple's native array framework).

A second backend alongside the PyTorch/MPS port (`mac_zonos2.py`). The whole
language-model forward + autoregressive decode runs in MLX (native to Apple
Silicon: unified memory, op fusion, lower dispatch overhead than routing
PyTorch through Metal). The DAC vocoder and the speaker encoder stay in
PyTorch (they run once, outside the hot loop).

Same architecture as the reference / the MPS port: GQA + QK-RMSNorm + per-head
temp + headwise gating, interleaved RoPE, dense + Sonic-MoE FFN with EDA router,
multi-codebook delay-pattern decode, 44.1 kHz DAC.

Usage:
    python mlx_zonos2.py --text "Hello world." --out out.wav
    python mlx_zonos2.py --text "Cloned." --voice default_voices/AmericanFemale.mp3 --out clone.wav
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import time

import mlx.core as mx
import numpy as np

# ---- Config (Zyphra/ZONOS2 params.json) ------------------------------------
N_LAYERS, DIM, HEAD_DIM, N_HEADS, N_KV_HEADS = 28, 2048, 128, 16, 4
INTERMEDIATE, ROPE_THETA, MAX_SEQLEN, RMS_EPS = 3072, 10000.0, 6144, 1e-5
N_CODEBOOKS, CODEBOOK_SIZE, AUDIO_VOCAB = 9, 1024, 1026
EOA_ID, AUDIO_PAD_ID, TEXT_VOCAB, LOSS_SOFTCAP = 1024, 1025, 519, 15.0
MOE_N_EXPERTS, MOE_START, MOE_END, SPECIAL_TOPK = 16, 3, 1, {26: 2}
SPEAKER_EMB_DIM, SPEAKER_LDA_DIM = 2048, 1024
QWEN3_SPEAKER_MODEL = "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B"
SPEAKING_RATE_BUCKETS, QUALITY_COUNTS = 8, [12, 12, 12, 8, 8, 8]
BACKGROUND_BUCKETS, ACCURATE_BUCKETS, LEGACY_SYMBOLS, BOS_ID, EOS_ID = 2, 1, 192, 2, 3
SILENCE_0_2S = [
    [568, 778, 338, 524, 967, 360, 728, 550, 90],
    [568, 778, 10, 674, 364, 981, 741, 378, 731],
    *[[568, 804, 10, 674, 364, 981, 568, 378, 731]] * 14,
    [568, 778, 721, 842, 264, 974, 989, 507, 308],
]


def is_moe_layer(i): return MOE_START <= i and (N_LAYERS - i) > MOE_END
def layer_topk(i): return SPECIAL_TOPK.get(i, 1)


# ---- Prompt construction (numpy) -------------------------------------------
def quality_token_id(feature_idx, bucket):
    base = TEXT_VOCAB - SPEAKING_RATE_BUCKETS - sum(QUALITY_COUNTS) - BACKGROUND_BUCKETS - ACCURATE_BUCKETS
    return base + SPEAKING_RATE_BUCKETS + sum(QUALITY_COUNTS[:feature_idx]) + bucket


def shear_np(rows, pad):
    T, C = rows.shape
    padded = np.full((C - 1 + T, C), pad, dtype=rows.dtype)
    padded[C - 1:] = rows
    idx = (C - 1) + np.arange(T)[:, None] - np.arange(C)[None, :]
    return np.take_along_axis(padded, idx, axis=0)


def shear_up_np(x, pad):
    H, W = x.shape[-2:]
    out = np.full(x.shape, pad, dtype=x.dtype)
    for j in range(W):
        if H > j:
            out[..., : H - j, j] = x[..., j:, j]
    return out


def build_prompt(text, with_speaker, trailing_silence_bucket=3):
    rows = []
    if with_speaker:
        rows.append([AUDIO_PAD_ID] * N_CODEBOOKS + [TEXT_VOCAB])
    if trailing_silence_bucket is not None:
        rows.append([AUDIO_PAD_ID] * N_CODEBOOKS + [quality_token_id(5, trailing_silence_bucket)])
    byte_ids = [BOS_ID, *(b + LEGACY_SYMBOLS for b in text.encode("utf-8")), EOS_ID]
    for tok in byte_ids:
        rows.append([AUDIO_PAD_ID] * N_CODEBOOKS + [tok])
    text_part = np.array(rows, dtype=np.int64)
    sil = shear_np(np.array(SILENCE_0_2S, dtype=np.int64), AUDIO_PAD_ID)
    sil = np.concatenate([sil, np.full((sil.shape[0], 1), TEXT_VOCAB, dtype=np.int64)], axis=1)
    return np.concatenate([text_part, sil], axis=0)


# ---- Speaker encoder (PyTorch, runs once) ----------------------------------
class SpeakerEncoder:
    TARGET_SR, N_FFT, HOP, WIN, N_MELS, F_MIN, F_MAX = 24_000, 1024, 256, 1024, 128, 0.0, 12_000.0

    def __init__(self, device="cpu"):
        import torch, torchaudio
        from transformers import AutoModel
        self.torch = torch
        self.device = device
        self.model = AutoModel.from_pretrained(QWEN3_SPEAKER_MODEL, trust_remote_code=True)
        self.model.to(device).eval().requires_grad_(False)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.TARGET_SR, n_fft=self.N_FFT, win_length=self.WIN, hop_length=self.HOP,
            f_min=self.F_MIN, f_max=self.F_MAX, n_mels=self.N_MELS, power=1.0, center=False,
            norm="slaney", mel_scale="slaney").to(device)

    def embed(self, wav, sr):
        import torch, torchaudio
        import torch.nn.functional as F
        with torch.inference_mode():
            wav = wav.mean(0, keepdim=True) if wav.ndim == 2 else wav.unsqueeze(0)
            wav = wav.to(self.device, torch.float32)
            if sr != self.TARGET_SR:
                wav = torchaudio.transforms.Resample(sr, self.TARGET_SR).to(self.device)(wav)
            pad = (self.N_FFT - self.HOP) // 2
            wav = F.pad(wav.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
            mel = torch.log(torch.clamp(self.mel(wav), min=1e-5)).transpose(1, 2)
            out = self.model(input_values=mel).last_hidden_state.to(torch.float32)
        return out.reshape(-1)[:SPEAKER_EMB_DIM].contiguous()


def get_speaker_embedding(voice_path, device="cpu"):
    cache = voice_path + ".zonos2spk.npy"
    if os.path.exists(cache):
        return np.load(cache).astype(np.float32)
    import torch, soundfile as sf
    wav_np, sr = sf.read(voice_path, dtype="float32", always_2d=True)
    print(f"[voice] encoding {voice_path} via Qwen3/ECAPA on {device} ...", flush=True)
    enc = SpeakerEncoder(device=device)
    emb = enc.embed(torch.from_numpy(wav_np.T), sr).cpu().numpy().astype(np.float32)
    np.save(cache, emb)
    del enc
    return emb


# ---- MLX helpers -----------------------------------------------------------
def lin(x, w, b=None):
    y = mx.matmul(x, w.swapaxes(-1, -2))
    return y + b if b is not None else y


def rmsnorm(x, weight, eps):
    dt = x.dtype
    x = x.astype(mx.float32)
    x = x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
    x = x.astype(dt)
    return x if weight is None else x * weight


def silu(x): return x * mx.sigmoid(x)
def gelu(x): return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))


# ---- Model -----------------------------------------------------------------
class Zonos2MLX:
    def __init__(self, ckpt_path, dtype=mx.bfloat16):
        import torch
        self.dtype = dtype
        t0 = time.time()
        print(f"[load] reading {ckpt_path} (dtype={dtype}) ...", flush=True)
        sd = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
        if "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]

        def to_mx(t, dt=dtype):
            return mx.array(t.float().numpy()).astype(dt)

        def g(key, dt=dtype):
            return to_mx(sd[key], dt)

        self.embed = [g(f"multi_embedder.embedders.{i}.weight") for i in range(N_CODEBOOKS + 1)]
        self.out_norm = g("out_norm.weight")
        self.multi_output = g("multi_output.weight")
        self.spk_lda_w = g("speaker_lda_projection.weight", mx.float32)
        self.spk_lda_b = g("speaker_lda_projection.bias", mx.float32)
        self.spk_w = g("speaker_projection.weight", mx.float32)
        self.spk_b = g("speaker_projection.bias", mx.float32)

        self.layers = []
        for i in range(N_LAYERS):
            p = f"layers.{i}."
            L = {
                "attn_norm": g(p + "attention_norm.weight"),
                "ffn_norm": g(p + "ffn_norm.weight"),
                "wq": g(p + "attention.wq.weight"),
                "wkv": to_mx(sd[p + "attention.wkv.weight"].reshape(2 * N_KV_HEADS * HEAD_DIM, DIM)),
                "wo": g(p + "attention.wo.weight"),
                "gater": g(p + "attention.gater.weight"),
                "temp": g(p + "attention.temp", mx.float32).abs().reshape(N_HEADS, 1),
                "is_moe": is_moe_layer(i),
            }
            if L["is_moe"]:
                import torch as _t
                w13 = sd[p + "feed_forward.experts.w13"]
                gate_up = _t.cat([w13[:, 0::2, :], w13[:, 1::2, :]], dim=1).contiguous()
                L["gate_up"] = to_mx(gate_up)
                L["down"] = g(p + "feed_forward.experts.w2")
                L["r_down_w"] = g(p + "feed_forward.router.down_proj.weight")
                L["r_down_b"] = g(p + "feed_forward.router.down_proj.bias")
                L["r_norm"] = g(p + "feed_forward.router.rmsnorm_eda.weight")
                L["r_m0w"] = g(p + "feed_forward.router.router_mlp.0.weight")
                L["r_m0b"] = g(p + "feed_forward.router.router_mlp.0.bias")
                L["r_m2w"] = g(p + "feed_forward.router.router_mlp.2.weight")
                L["r_m2b"] = g(p + "feed_forward.router.router_mlp.2.bias")
                L["r_m4w"] = g(p + "feed_forward.router.router_mlp.4.weight")
                L["bias"] = g(p + "feed_forward.router.balancing_biases", mx.float32)
                rss_key = p + "feed_forward.router.router_states_scale"
                L["use_eda"] = (i != MOE_START) and (rss_key in sd)
                L["rss"] = g(rss_key) if rss_key in sd else None
                L["topk"] = layer_topk(i)
            else:
                L["w_in"] = to_mx(sd[p + "feed_forward.w_in.weight"].reshape(2 * INTERMEDIATE, DIM))
                L["w_out"] = g(p + "feed_forward.w_out.weight")
            self.layers.append(L)
            mx.eval([v for v in L.values() if isinstance(v, mx.array)])

        inv = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2) / HEAD_DIM))
        freqs = np.outer(np.arange(MAX_SEQLEN), inv)
        self.cos = mx.array(np.cos(freqs).astype(np.float32))
        self.sin = mx.array(np.sin(freqs).astype(np.float32))
        self.k_buf = [mx.zeros((MAX_SEQLEN, N_KV_HEADS, HEAD_DIM), dtype=dtype) for _ in range(N_LAYERS)]
        self.v_buf = [mx.zeros((MAX_SEQLEN, N_KV_HEADS, HEAD_DIM), dtype=dtype) for _ in range(N_LAYERS)]
        self.cache_len = 0
        mx.eval(self.cos, self.sin, *self.embed, self.multi_output)
        del sd
        print(f"[load] done in {time.time()-t0:.1f}s on {mx.default_device()}", flush=True)

    def _rope(self, x, positions):
        dt = x.dtype
        cos = self.cos[positions][:, None, :]
        sin = self.sin[positions][:, None, :]
        x = x.astype(mx.float32).reshape(x.shape[0], x.shape[1], HEAD_DIM // 2, 2)
        x0, x1 = x[..., 0], x[..., 1]
        out = mx.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], axis=-1)
        return out.reshape(out.shape[0], out.shape[1], HEAD_DIM).astype(dt)

    def _attention(self, L, x, positions, li, is_prefill):
        T = x.shape[0]
        gate = mx.sigmoid(lin(x, L["gater"]))
        q = lin(x, L["wq"]).reshape(T, N_HEADS, HEAD_DIM)
        kv = lin(x, L["wkv"])
        k = kv[:, : N_KV_HEADS * HEAD_DIM].reshape(T, N_KV_HEADS, HEAD_DIM)
        v = kv[:, N_KV_HEADS * HEAD_DIM:].reshape(T, N_KV_HEADS, HEAD_DIM)

        q = rmsnorm(q, None, 1e-6) * L["temp"].astype(q.dtype)
        k = rmsnorm(k, None, 1e-6)
        q = self._rope(q, positions)
        k = self._rope(k, positions)

        s = self.cache_len
        self.k_buf[li][s:s + T] = k
        self.v_buf[li][s:s + T] = v
        k_all = self.k_buf[li][:s + T]
        v_all = self.v_buf[li][:s + T]

        rep = N_HEADS // N_KV_HEADS
        kh = mx.repeat(k_all, rep, axis=1).transpose(1, 0, 2)[None]   # (1,16,Tk,128)
        vh = mx.repeat(v_all, rep, axis=1).transpose(1, 0, 2)[None]
        qh = q.transpose(1, 0, 2)[None]                                # (1,16,T,128)
        mask = "causal" if is_prefill else None
        o = mx.fast.scaled_dot_product_attention(qh, kh, vh, scale=HEAD_DIM ** -0.5, mask=mask)
        o = o[0].transpose(1, 0, 2) * gate[..., None]                 # (T,16,128)
        return lin(o.reshape(T, N_HEADS * HEAD_DIM), L["wo"])

    def _dense_ffn(self, L, x):
        h = lin(x, L["w_in"])
        return lin(h[:, :INTERMEDIATE] * silu(h[:, INTERMEDIATE:]), L["w_out"])

    def _moe_ffn(self, L, x, router_states):
        T = x.shape[0]
        r = lin(x, L["r_down_w"], L["r_down_b"])
        if L["use_eda"] and router_states is not None:
            r = r + router_states * L["rss"]
        router_states_next = r
        r = rmsnorm(r, L["r_norm"], RMS_EPS)
        m = gelu(lin(r, L["r_m0w"], L["r_m0b"]))
        m = gelu(lin(m, L["r_m2w"], L["r_m2b"]))
        prob = mx.softmax(lin(m, L["r_m4w"]).astype(mx.float32), axis=-1, precise=True)
        scores = prob + L["bias"]                                    # legacy balancing bias
        k = L["topk"]
        inds = mx.argpartition(scores, kth=-k, axis=-1)[..., -k:]     # (T, k) chosen experts
        weight = mx.take_along_axis(prob, inds, axis=-1).astype(x.dtype)  # (T, k) — no renorm

        # Fused grouped MoE via mx.gather_mm (the idiomatic MLX MoE primitive):
        # picks each token's expert weights by index without materializing them.
        xe = mx.expand_dims(x, (-2, -3))                             # (T,1,1,dim)
        gu = mx.gather_mm(xe, L["gate_up"].swapaxes(-1, -2), rhs_indices=inds)  # (T,k,1,2*inter)
        act = silu(gu[..., :INTERMEDIATE]) * gu[..., INTERMEDIATE:]
        y = mx.gather_mm(act, L["down"].swapaxes(-1, -2), rhs_indices=inds)     # (T,k,1,dim)
        out = (y.squeeze(-2) * weight[..., None]).sum(axis=-2)       # (T, dim)
        return out, router_states_next

    def forward(self, input_ids_np, positions_np, is_prefill, spk_emb=None, spk_pos=None):
        ids = mx.array(input_ids_np)
        pos = mx.array(positions_np)
        x = self.embed[0][ids[:, 0]]
        for i in range(1, N_CODEBOOKS + 1):
            x = x + self.embed[i][ids[:, i]]
        if spk_emb is not None and spk_pos is not None:
            lda = lin(spk_emb, self.spk_lda_w, self.spk_lda_b)
            proj = lin(lda, self.spk_w, self.spk_b).astype(x.dtype)
            x[spk_pos] = proj
        x = rmsnorm(x, None, RMS_EPS)

        residual = None
        router_states = None
        for li, L in enumerate(self.layers):
            if residual is None:
                normed = rmsnorm(x, L["attn_norm"], RMS_EPS)
                residual = x
            else:
                residual = residual + x
                normed = rmsnorm(residual, L["attn_norm"], RMS_EPS)
            residual = residual + self._attention(L, normed, pos, li, is_prefill)
            normed2 = rmsnorm(residual, L["ffn_norm"], RMS_EPS)
            if L["is_moe"]:
                x, router_states = self._moe_ffn(L, normed2, router_states)
            else:
                x, router_states = self._dense_ffn(L, normed2), None
        self.cache_len += input_ids_np.shape[0]
        h = rmsnorm(residual + x, self.out_norm, RMS_EPS)
        logits = lin(h, self.multi_output).reshape(h.shape[0], N_CODEBOOKS, AUDIO_VOCAB)
        return LOSS_SOFTCAP * mx.tanh(logits / LOSS_SOFTCAP)

    def generate(self, text, voice_emb=None, max_tokens=1024, temperature=1.15, topk=106,
                 min_p=0.18, rep_window=50, rep_penalty=1.2, rep_codebooks=8, seed=42, verbose=True):
        rng = np.random.default_rng(seed)
        self.cache_len = 0
        prompt = build_prompt(text, with_speaker=voice_emb is not None)
        spk_emb = mx.array(voice_emb.astype(np.float32)) if voice_emb is not None else None
        spk_pos = 0 if voice_emb is not None else None

        P = prompt.shape[0]
        t0 = time.time()
        logits = self.forward(prompt, np.arange(P), True, spk_emb, spk_pos)[-1]
        mx.eval(logits)

        frames, eos_frame, eos_countdown, cur = [], None, 0, P
        for step in range(max_tokens):
            tok = self._sample(np.array(logits.astype(mx.float32)), frames, temperature, topk,
                               min_p, rep_window, rep_penalty, rep_codebooks, rng)
            frames.append(tok)
            if eos_frame is None and any(t == EOA_ID for t in tok):
                eos_frame = max(0, step - max(c for c, t in enumerate(tok) if t == EOA_ID))
                eos_countdown = N_CODEBOOKS + 1
            if eos_frame is not None:
                eos_countdown -= 1
                if eos_countdown <= 0:
                    break
            row = np.array([tok + [TEXT_VOCAB]], dtype=np.int64)
            logits = self.forward(row, np.array([cur]), False)[-1]
            mx.eval(logits)
            cur += 1
            if verbose and (step + 1) % 50 == 0:
                print(f"  step {step+1} ({(step+1)/(time.time()-t0):.1f} frames/s)", flush=True)

        dt = time.time() - t0
        print(f"[gen] {len(frames)} frames, eos_frame={eos_frame} in {dt:.1f}s "
              f"({len(frames)/dt:.1f} frames/s)", flush=True)
        return frames, eos_frame

    def _sample(self, lg, frames, temperature, topk, min_p, rep_window, rep_penalty, rep_codebooks, rng):
        lg = lg.astype(np.float32).copy()                            # (9, V)
        if rep_penalty > 1.0 and rep_window > 0 and frames:
            hist = np.array(frames[-rep_window:]).T                  # (9, w)
            for c in range(min(rep_codebooks, N_CODEBOOKS)):
                ids = hist[c]
                ids = np.unique(ids[(ids >= 0) & (ids < CODEBOOK_SIZE)])
                if ids.size:
                    row = lg[c]
                    row[ids] = np.where(row[ids] > 0, row[ids] / rep_penalty, row[ids] * rep_penalty)
        if temperature <= 0:
            return lg.argmax(-1).tolist()
        lg = lg / max(temperature, 1e-6)
        if 0 < topk < lg.shape[-1]:
            kth = np.partition(lg, -topk, axis=-1)[:, -topk][:, None]
            lg = np.where(lg < kth, -np.inf, lg)
        probs = np.exp(lg - lg.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        if min_p > 0:
            probs = np.where(probs < min_p * probs.max(-1, keepdims=True), 0.0, probs)
            probs /= np.clip(probs.sum(-1, keepdims=True), 1e-8, None)
        # Gumbel-max categorical sampling per codebook
        g = -np.log(-np.log(np.clip(rng.random(probs.shape), 1e-12, 1.0)))
        return (np.log(np.clip(probs, 1e-12, None)) + g).argmax(-1).tolist()


# ---- Vocoder (PyTorch DAC) -------------------------------------------------
def decode_to_audio(frames, eos_frame, device="mps"):
    import torch, dac
    codes = shear_up_np(np.array(frames, dtype=np.int64), AUDIO_PAD_ID)
    if eos_frame is not None:
        codes = codes[: max(0, eos_frame)]
    if codes.size == 0:
        return None, 44100
    codes = np.clip(codes, None, CODEBOOK_SIZE - 1)
    codes = torch.from_numpy(codes).to(device).unsqueeze(0).permute(0, 2, 1)   # (1,9,F)
    model = dac.DAC.load(dac.utils.download(model_type="44khz")).eval().to(device)
    with torch.no_grad():
        z = model.quantizer.from_codes(codes)[0]
        audio = model.decode(z).float().squeeze(1).squeeze(0).cpu().numpy()
    return audio, 44100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="Hello from Zonos 2, running on Apple Silicon with M.L.X.")
    ap.add_argument("--voice", default=None)
    ap.add_argument("--out", default="output_mlx.wav")
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=1.15)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--speaker-device", default="cpu")
    ap.add_argument("--dac-device", default="mps")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    dtype = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}[args.dtype]
    ckpt = args.ckpt or glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--Zyphra--ZONOS2/snapshots/*/model.pth"))[0]

    voice_emb = get_speaker_embedding(args.voice, args.speaker_device) if args.voice else None
    tts = Zonos2MLX(ckpt, dtype=dtype)
    print(f"[gen] text: {args.text!r}  voice: {args.voice or '(default)'}", flush=True)
    frames, eos_frame = tts.generate(args.text, voice_emb=voice_emb, max_tokens=args.max_tokens,
                                     temperature=args.temperature, seed=args.seed)
    audio, sr = decode_to_audio(frames, eos_frame, device=args.dac_device)
    if audio is None:
        print("No audio generated.")
        return
    import soundfile as sf
    sf.write(args.out, audio, sr)
    print(f"[done] wrote {args.out}  ({len(audio)/sr:.2f}s @ {sr} Hz)", flush=True)


if __name__ == "__main__":
    main()
