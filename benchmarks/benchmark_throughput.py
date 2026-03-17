"""
Throughput benchmarking suite for TinyFormer-USB.

Measures tokens per second for the reference NumPy inference implementation
at different model sizes, sequence lengths, and quantisation settings.

This benchmark runs entirely on the host CPU and serves as a software
baseline. The target USB hardware (NPU/FPGA) must achieve ≥ 20 000 tok/s.

Usage
-----
    python benchmarks/benchmark_throughput.py
    python benchmarks/benchmark_throughput.py --n_tokens 100 --n_warmup 5
"""

import argparse
import time

import numpy as np

from src.model.tiny_transformer import (
    TinyFormerConfig,
    TinyFormerInference,
    ModelWeights,
    random_weights,
    quantize_int4,
    dequantize_int4,
)
from src.model.tokenizer import build_demo_tokenizer, BOS_ID


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _make_model(
    n_layers: int = 12,
    d_model: int = 512,
    n_heads: int = 8,
    d_ff: int = 2048,
    vocab_size: int = 16_384,
    seed: int = 42,
) -> tuple[TinyFormerConfig, TinyFormerInference]:
    config = TinyFormerConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
    )
    weights = random_weights(config, seed=seed)
    model = TinyFormerInference(config, weights)
    return config, model


def measure_token_throughput(
    model: TinyFormerInference,
    n_tokens: int = 50,
    n_warmup: int = 3,
) -> dict:
    """
    Measure inference throughput by generating ``n_tokens`` auto-regressively.

    Returns
    -------
    dict with keys: tokens_per_second, latency_ms_per_token, total_seconds
    """
    model.reset()

    # Warmup
    for _ in range(n_warmup):
        model.reset()
        model.forward_one_token(BOS_ID)

    model.reset()
    model.forward_one_token(BOS_ID)   # seed with BOS

    token_id = BOS_ID
    start = time.perf_counter()
    for _ in range(n_tokens):
        logits = model.forward_one_token(token_id)
        token_id = int(np.argmax(logits))  # greedy decode
    elapsed = time.perf_counter() - start

    tps   = n_tokens / elapsed
    ms_pt = elapsed / n_tokens * 1_000

    return {
        "tokens_per_second":     round(tps, 1),
        "latency_ms_per_token":  round(ms_pt, 2),
        "total_seconds":         round(elapsed, 3),
        "n_tokens":              n_tokens,
    }


def measure_quantization_speed(n_rows: int = 512, n_cols: int = 512) -> dict:
    """
    Measure INT4 quantise / dequantise speed on a sample weight matrix.
    """
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((n_rows, n_cols)).astype(np.float32)

    # Quantise
    t0 = time.perf_counter()
    packed, scale, zero = quantize_int4(weight)
    t_quant = time.perf_counter() - t0

    # Dequantise
    t0 = time.perf_counter()
    reconstructed = dequantize_int4(packed, scale, zero)
    t_dequant = time.perf_counter() - t0

    # Error
    max_err = float(np.max(np.abs(weight - reconstructed)))
    rms_err = float(np.sqrt(np.mean((weight - reconstructed) ** 2)))

    return {
        "matrix_shape":           (n_rows, n_cols),
        "quant_time_ms":          round(t_quant * 1_000, 3),
        "dequant_time_ms":        round(t_dequant * 1_000, 3),
        "max_absolute_error":     round(max_err, 6),
        "rms_error":              round(rms_err, 6),
        "size_fp32_bytes":        weight.nbytes,
        "size_int4_bytes":        packed.nbytes + scale.nbytes,
        "compression_ratio":      round(weight.nbytes / (packed.nbytes + scale.nbytes), 2),
    }


# ---------------------------------------------------------------------------
# Model size sweep
# ---------------------------------------------------------------------------

MODEL_CONFIGS = [
    # (label, n_layers, d_model, n_heads, d_ff)
    ("1M",   2,   128,  4,   512),
    ("5M",   6,   256,  4,   1024),
    ("10M",  8,   384,  6,   1536),
    ("50M",  12,  512,  8,   2048),
]


def run_benchmark(n_tokens: int = 30, n_warmup: int = 2) -> None:
    """Run the full benchmark suite and print results."""
    print("\n" + "=" * 68)
    print("TinyFormer-USB – Software Throughput Benchmark (NumPy baseline)")
    print("=" * 68)
    print(
        "(Note: Host CPU / NumPy is a reference baseline. "
        "Target USB hardware is expected to be 100–1000× faster.)\n"
    )

    # 1. Quantisation accuracy
    print("INT4 Quantisation Accuracy & Speed")
    print("-" * 40)
    for shape in [(256, 512), (512, 512), (512, 2048)]:
        r = measure_quantization_speed(*shape)
        print(
            f"  {shape[0]}×{shape[1]:4}  "
            f"quant={r['quant_time_ms']:6.2f}ms  "
            f"dequant={r['dequant_time_ms']:6.2f}ms  "
            f"RMS err={r['rms_error']:.5f}  "
            f"compression={r['compression_ratio']}×"
        )
    print()

    # 2. Token throughput by model size
    print(f"Token Generation Throughput  (n_tokens={n_tokens})")
    print("-" * 68)
    print(f"  {'Model':>8}  {'Params':>8}  {'TPS':>10}  {'ms/tok':>8}  {'Target':>10}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}")

    for label, nl, dm, nh, dff in MODEL_CONFIGS:
        config, model = _make_model(
            n_layers=nl, d_model=dm, n_heads=nh, d_ff=dff,
        )
        r = measure_token_throughput(model, n_tokens=n_tokens, n_warmup=n_warmup)
        params_m = config.n_params / 1e6
        target = "≥ 20 kt/s needed on HW"
        print(
            f"  {label:>8}  {params_m:>7.1f}M  "
            f"{r['tokens_per_second']:>10,.0f}  "
            f"{r['latency_ms_per_token']:>8.2f}  "
            f"{target}"
        )

    print()

    # 3. Flash size estimate
    print("On-device Flash Storage Requirements (INT4 quantised)")
    print("-" * 68)
    print(f"  {'Model':>8}  {'Params':>8}  {'FP32 size':>12}  {'INT4 size':>12}  {'Fits 128MB?':>12}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}")
    for label, nl, dm, nh, dff in MODEL_CONFIGS:
        config = TinyFormerConfig(
            vocab_size=16_384, d_model=dm, n_heads=nh, n_layers=nl, d_ff=dff
        )
        sizes = config.weight_size_bytes
        fp32_mb = sizes["fp32"] / 1024 / 1024
        int4_mb = sizes["int4"] / 1024 / 1024
        fits    = "✓" if int4_mb < 100 else "✗"
        print(
            f"  {label:>8}  {config.n_params/1e6:>7.1f}M  "
            f"{fp32_mb:>10.1f} MB  "
            f"{int4_mb:>10.1f} MB  "
            f"{fits:>12}"
        )
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TinyFormer-USB throughput benchmark"
    )
    parser.add_argument(
        "--n_tokens", type=int, default=30,
        help="Number of tokens to generate per run (default: 30)"
    )
    parser.add_argument(
        "--n_warmup", type=int, default=2,
        help="Number of warmup iterations (default: 2)"
    )
    args = parser.parse_args()
    run_benchmark(n_tokens=args.n_tokens, n_warmup=args.n_warmup)
