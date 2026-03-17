"""
TinyFormer-USB: Ultra-compact decoder-only Transformer designed for
USB-form-factor inference devices targeting ≥ 20 000 tokens per second.

Architecture:
  - Decoder-only (GPT-style)
  - 50 M parameters (target)
  - d_model=512, n_heads=8, n_layers=12, d_ff=2048
  - Vocabulary size: 16 384
  - Context window: 512 tokens
  - Designed for INT4 weight / INT8 activation (W4A8) deployment
"""

import math
import struct
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TinyFormerConfig:
    """All hyper-parameters for the TinyFormer-USB model."""

    vocab_size: int = 16_384
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 12
    d_ff: int = 2_048
    max_seq_len: int = 512
    dropout: float = 0.0        # zero for inference-only deployment
    # head dimension (derived)
    head_dim: int = field(init=False)

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, (
            "d_model must be divisible by n_heads"
        )
        self.head_dim = self.d_model // self.n_heads

    @property
    def n_params(self) -> int:
        """Approximate parameter count (excludes biases)."""
        embed = self.vocab_size * self.d_model
        attn = 4 * self.d_model * self.d_model   # Q, K, V, O projections
        ff = 2 * self.d_model * self.d_ff          # up + down projections
        return embed + self.n_layers * (attn + ff)

    @property
    def weight_size_bytes(self) -> dict:
        """Approximate on-disk sizes per quantisation precision."""
        params = self.n_params
        return {
            "fp32": params * 4,
            "fp16": params * 2,
            "int8": params * 1,
            "int4": params // 2,
            "int2": params // 4,
        }


# ---------------------------------------------------------------------------
# Quantisation helpers (pure NumPy, hardware-portable)
# ---------------------------------------------------------------------------

def quantize_int4(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Symmetric per-row INT4 quantisation.

    Returns
    -------
    packed : uint8 ndarray, shape (*weight.shape[:-1], weight.shape[-1] // 2)
        Two INT4 values packed per byte (little-endian nibble order).
    scale  : float32 ndarray, shape (*weight.shape[:-1],)
        Per-row scale factor (max_abs / 7).
    zero   : float32 ndarray  (all zeros for symmetric quant)
    """
    w = weight.astype(np.float32)
    max_abs = np.max(np.abs(w), axis=-1, keepdims=True).clip(min=1e-8)
    scale = (max_abs / 7.0).squeeze(-1).astype(np.float32)

    # Quantise to [-7, 7]
    q = np.round(w / max_abs * 7.0).clip(-7, 7).astype(np.int8)
    q_unsigned = (q + 8).astype(np.uint8)          # shift to [1, 15] (0 reserved)

    # Pack two nibbles per byte
    flat = q_unsigned.reshape(-1, q_unsigned.shape[-1])
    assert flat.shape[-1] % 2 == 0, "last dimension must be even"
    lo = flat[..., 0::2] & 0x0F
    hi = flat[..., 1::2] & 0x0F
    packed = (lo | (hi << 4)).reshape(*weight.shape[:-1], weight.shape[-1] // 2)

    zero = np.zeros_like(scale)
    return packed, scale, zero


def dequantize_int4(
    packed: np.ndarray,
    scale: np.ndarray,
    zero: np.ndarray,
) -> np.ndarray:
    """Reconstruct float32 weights from INT4-packed representation."""
    lo = (packed & 0x0F).astype(np.int8)
    hi = ((packed >> 4) & 0x0F).astype(np.int8)

    interleaved = np.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=np.int8)
    interleaved[..., 0::2] = lo
    interleaved[..., 1::2] = hi

    q = interleaved.astype(np.float32) - 8.0   # un-shift from [1,15] → [-7,7]
    return q * scale[..., np.newaxis]


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def silu(x: np.ndarray) -> np.ndarray:
    """SiLU / Swish activation: x * sigmoid(x)."""
    return x / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Root-mean-square layer normalisation (no bias)."""
    rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    return x / rms * weight


# ---------------------------------------------------------------------------
# Attention (single head, then multi-head)
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    q: np.ndarray,   # (seq, head_dim)
    k: np.ndarray,   # (seq, head_dim)
    v: np.ndarray,   # (seq, head_dim)
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Masked scaled dot-product attention for a single head."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = q @ k.T * scale                          # (seq, seq)
    if mask is not None:
        scores = scores + mask
    attn = softmax(scores, axis=-1)                   # (seq, seq)
    return attn @ v                                   # (seq, head_dim)


# ---------------------------------------------------------------------------
# Transformer layer weights container
# ---------------------------------------------------------------------------

@dataclass
class TransformerLayerWeights:
    """
    All weights for a single Transformer decoder layer.
    Stored as float32 for reference; will be quantised to INT4 for deployment.
    """
    # Attention
    w_q: np.ndarray        # (d_model, d_model)
    w_k: np.ndarray        # (d_model, d_model)
    w_v: np.ndarray        # (d_model, d_model)
    w_o: np.ndarray        # (d_model, d_model)
    # Feed-forward (SwiGLU variant: two up-projections + one down)
    w_ff_up1: np.ndarray   # (d_model, d_ff)
    w_ff_up2: np.ndarray   # (d_model, d_ff)  gate
    w_ff_down: np.ndarray  # (d_ff, d_model)
    # RMS norms
    norm1_weight: np.ndarray  # (d_model,)
    norm2_weight: np.ndarray  # (d_model,)


@dataclass
class ModelWeights:
    """All weights for the TinyFormer-USB model."""
    token_embed: np.ndarray            # (vocab_size, d_model)
    layers: list[TransformerLayerWeights]
    final_norm_weight: np.ndarray      # (d_model,)
    lm_head: np.ndarray                # (d_model, vocab_size)  (often = token_embed.T)


# ---------------------------------------------------------------------------
# Inference: single forward pass (autoregressive, one token at a time)
# ---------------------------------------------------------------------------

class TinyFormerInference:
    """
    Pure-NumPy reference implementation of TinyFormer-USB inference.

    This is the *software specification* that mirrors what the on-device
    firmware / FPGA logic must implement.  It is intentionally not optimised
    for Python speed; for benchmarking use the optimised C/CUDA backends.
    """

    def __init__(self, config: TinyFormerConfig, weights: ModelWeights) -> None:
        self.cfg = config
        self.w = weights

        # KV-cache: list of (k_cache, v_cache) per layer
        # Shape: (max_seq_len, n_heads, head_dim)
        self._reset_kv_cache()

    def _reset_kv_cache(self) -> None:
        c = self.cfg
        empty = np.zeros((c.max_seq_len, c.n_heads, c.head_dim), dtype=np.float32)
        self.kv_cache: list[tuple[np.ndarray, np.ndarray]] = [
            (empty.copy(), empty.copy()) for _ in range(c.n_layers)
        ]
        self.seq_len: int = 0

    def _attention_layer(
        self,
        x: np.ndarray,                               # (1, d_model)  current token
        layer_idx: int,
    ) -> np.ndarray:
        cfg = self.cfg
        wl = self.w.layers[layer_idx]
        pos = self.seq_len - 1  # 0-indexed position of current token

        # Project Q/K/V
        q = x @ wl.w_q   # (1, d_model)
        k = x @ wl.w_k
        v = x @ wl.w_v

        # Reshape to multi-head: (1, n_heads, head_dim)
        q = q.reshape(1, cfg.n_heads, cfg.head_dim)
        k = k.reshape(1, cfg.n_heads, cfg.head_dim)
        v = v.reshape(1, cfg.n_heads, cfg.head_dim)

        # Store in KV-cache
        k_cache, v_cache = self.kv_cache[layer_idx]
        k_cache[pos] = k[0]
        v_cache[pos] = v[0]

        # Attend over all past tokens (pos+1 tokens in total)
        output_heads = []
        for h in range(cfg.n_heads):
            q_h = q[0, h]                        # (head_dim,)
            k_h = k_cache[:pos + 1, h]           # (pos+1, head_dim)
            v_h = v_cache[:pos + 1, h]           # (pos+1, head_dim)
            out_h = scaled_dot_product_attention(
                q_h[np.newaxis],  # (1, head_dim)
                k_h,              # (pos+1, head_dim)
                v_h,              # (pos+1, head_dim)
            )
            output_heads.append(out_h[0])         # (head_dim,)

        # Concatenate heads and project
        attn_out = np.concatenate(output_heads, axis=-1)[np.newaxis]  # (1, d_model)
        return attn_out @ wl.w_o                                        # (1, d_model)

    def _ff_layer(self, x: np.ndarray, layer_idx: int) -> np.ndarray:
        """SwiGLU feed-forward: down(silu(up1(x)) * up2(x))."""
        wl = self.w.layers[layer_idx]
        gate = silu(x @ wl.w_ff_up1)          # (1, d_ff)
        up   = x @ wl.w_ff_up2                # (1, d_ff)
        return (gate * up) @ wl.w_ff_down     # (1, d_model)

    def forward_one_token(self, token_id: int) -> np.ndarray:
        """
        Run one forward step and return the logit vector over the vocabulary.
        Updates the internal KV-cache.

        Parameters
        ----------
        token_id : int  – index in [0, vocab_size)

        Returns
        -------
        logits : float32 ndarray of shape (vocab_size,)
        """
        if self.seq_len >= self.cfg.max_seq_len:
            raise RuntimeError(
                f"Context window exceeded ({self.cfg.max_seq_len} tokens). "
                "Call reset_kv_cache() to start a new sequence."
            )

        # Token embedding
        x = self.w.token_embed[token_id][np.newaxis]  # (1, d_model)
        self.seq_len += 1

        # Transformer layers
        for i in range(self.cfg.n_layers):
            wl = self.w.layers[i]
            # Pre-norm attention
            x = x + self._attention_layer(rms_norm(x, wl.norm1_weight), i)
            # Pre-norm feed-forward
            x = x + self._ff_layer(rms_norm(x, wl.norm2_weight), i)

        # Final norm + LM head
        x = rms_norm(x, self.w.final_norm_weight)   # (1, d_model)
        logits = (x @ self.w.lm_head)[0]             # (vocab_size,)
        return logits

    def reset(self) -> None:
        """Clear KV-cache for a new conversation."""
        self._reset_kv_cache()


# ---------------------------------------------------------------------------
# Weight initialisation (random, for unit tests and benchmarking)
# ---------------------------------------------------------------------------

def random_weights(config: TinyFormerConfig, seed: int = 42) -> ModelWeights:
    """
    Generate random float32 weights matching TinyFormer-USB dimensions.
    Used for benchmarking and unit tests only.
    """
    rng = np.random.default_rng(seed)
    c = config

    def randn(*shape: int) -> np.ndarray:
        return rng.standard_normal(shape).astype(np.float32) * 0.02

    def ones(*shape: int) -> np.ndarray:
        return np.ones(shape, dtype=np.float32)

    token_embed = randn(c.vocab_size, c.d_model)
    layers = []
    for _ in range(c.n_layers):
        layers.append(TransformerLayerWeights(
            w_q=randn(c.d_model, c.d_model),
            w_k=randn(c.d_model, c.d_model),
            w_v=randn(c.d_model, c.d_model),
            w_o=randn(c.d_model, c.d_model),
            w_ff_up1=randn(c.d_model, c.d_ff),
            w_ff_up2=randn(c.d_model, c.d_ff),
            w_ff_down=randn(c.d_ff, c.d_model),
            norm1_weight=ones(c.d_model),
            norm2_weight=ones(c.d_model),
        ))

    return ModelWeights(
        token_embed=token_embed,
        layers=layers,
        final_norm_weight=ones(c.d_model),
        lm_head=token_embed.T.copy(),   # weight tying
    )


# ---------------------------------------------------------------------------
# Weight serialisation (binary format for on-device Flash)
# ---------------------------------------------------------------------------

MAGIC = b"TINYF001"          # 8-byte magic number for the binary weight file
HEADER_FMT = "!8sII"         # magic(8), n_layers(4), vocab_size(4)
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def serialize_weights_int4(config: TinyFormerConfig, weights: ModelWeights) -> bytes:
    """
    Serialise model weights to INT4 packed binary format suitable for
    flashing to SPI NOR or eMMC on the USB device.

    Layout
    ------
    [HEADER]
    [token_embed INT4 packed]      (vocab_size × d_model / 2 bytes)
    [token_embed scale]            (vocab_size × 4 bytes)
    For each layer:
      [w_q, w_k, w_v, w_o INT4 + scales]
      [w_ff_up1, w_ff_up2, w_ff_down INT4 + scales]
      [norm1_weight float32]       (d_model × 4 bytes)
      [norm2_weight float32]       (d_model × 4 bytes)
    [final_norm_weight float32]
    """
    buf = bytearray()

    header = struct.pack(HEADER_FMT, MAGIC, config.n_layers, config.vocab_size)
    buf.extend(header)

    def pack_matrix(w: np.ndarray) -> bytes:
        packed, scale, _ = quantize_int4(w)
        return packed.tobytes() + scale.tobytes()

    # Embedding
    buf.extend(pack_matrix(weights.token_embed))

    # Layers
    for layer in weights.layers:
        for mat in (layer.w_q, layer.w_k, layer.w_v, layer.w_o,
                    layer.w_ff_up1, layer.w_ff_up2, layer.w_ff_down):
            buf.extend(pack_matrix(mat))
        buf.extend(layer.norm1_weight.astype(np.float32).tobytes())
        buf.extend(layer.norm2_weight.astype(np.float32).tobytes())

    # Final norm (not quantised – cheap)
    buf.extend(weights.final_norm_weight.astype(np.float32).tobytes())

    return bytes(buf)
