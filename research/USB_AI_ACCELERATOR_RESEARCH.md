# USB AI Accelerator – Research & Feasibility Study

## Goal

Design and prototype a **USB-form-factor AI inference device** that can deliver
**≥ 20 000 tokens per second (20 kt/s)** for a language model whose weights,
tokenizer, and runtime are all **hardcoded** on the device itself (no host-side
model files required).

---

## 1. Why USB Form Factor?

| Property | Value |
|---|---|
| Physical envelope | ~60 × 20 × 10 mm (standard USB-A dongle) |
| Power budget (USB 3.2 Gen 2) | ≤ 4.5 W |
| Host interface | USB 3.2 Gen 2 – up to 10 Gbps (1.2 GB/s) |
| Target use-cases | Air-gapped inference, edge deployment, secure enclaves |

---

## 2. Throughput Target Analysis

### 2.1  Token generation bottleneck

Auto-regressive token generation is **memory-bandwidth-bound**, not
compute-bound.  Each new token requires reading every weight in the model once.

```
Minimum bandwidth = #parameters × bytes_per_weight × tokens_per_second
```

| Quantisation | Bytes / param | Max params for 20 kt/s @ 50 GB/s BW |
|---|---|---|
| FP16 | 2 | ~1.25 M |
| INT8 | 1 | ~2.5 M |
| INT4 | 0.5 | ~5 M |
| INT2 | 0.25 | ~10 M |

> **Key insight**: to hit 20 kt/s within the 4.5 W envelope we need a model
> with ≤ **5 M INT4 parameters** given 50 GB/s on-package LPDDR5 bandwidth,
> OR a dedicated FPGA/ASIC with ≥ 200 GB/s on-chip SRAM for larger models.
> Larger models (50 M+) remain feasible but require custom silicon with HBM.

### 2.2  Compute requirement

For a 50 M-parameter transformer with INT4 weights:

```
FLOPs / token ≈ 2 × 50 M = 100 MFLOP
At 20 000 tok/s → 2 TFLOPS INT4 (= 0.5 TFLOPS FP32 equivalent)
```

This is achievable with custom silicon or a mid-range FPGA in 2025.

---

## 3. Hardware Options

### Option A – FPGA-based (prototype / research)

| Component | Part | Notes |
|---|---|---|
| FPGA | Xilinx Spartan-7 XC7S50 | ~52k LUTs, fits USB dongle PCB |
| SRAM | 4 × 512 KB SRAM (ISSI IS61WV51216) | 2 MB on-chip buffer |
| Flash | SPI NOR 128 MB | Weight storage |
| USB controller | FTDI FT601 (USB 3.0 ↔ FIFO) | Host ↔ FPGA bridge |
| Power | USB VBUS → 1.0 V / 1.8 V / 3.3 V LDOs | < 2 W |

**Estimated throughput**: 5–15 kt/s (INT4 @ ~200 GOPS for Spartan-7)

### Option B – NPU ASIC (production target)

| Component | Part | Notes |
|---|---|---|
| NPU | Hailo-8L or custom 7 nm ASIC | 13–26 TOPS INT8 |
| LPDDR5 | 2–4 GB | High-bandwidth on-package RAM |
| Flash (firmware) | eMMC 32 GB | Model weights, tokenizer |
| USB controller | USB 3.2 Gen 2 native | 10 Gbps |
| Power | < 3 W active (Hailo-8L: 2.5 W) | |

**Estimated throughput**: 20–50 kt/s (INT4 quantised 100 M-param model)

### Option C – M.2 / USB-C with discrete GPU tile (high-end)

Use an ultra-low-power GPU tile (e.g., Intel Arc A-series entry, AMD 890M
iGPU in M.2 module) with full CUDA/ROCm support for flexibility.

**Estimated throughput**: 50–200 kt/s — exceeds target but larger form factor.

---

## 4. Software Stack (Hardcoded on Device)

```
┌─────────────────────────────────────────────────────┐
│                   HOST COMPUTER                     │
│  USB Host Driver (libusb / WinUSB / IOKit)          │
└──────────────────┬──────────────────────────────────┘
                   │ USB 3.2 (token stream)
┌──────────────────▼──────────────────────────────────┐
│               USB DEVICE FIRMWARE                   │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │  Tokenizer │  │  KV-Cache    │  │  Sampler    │  │
│  │  (BPE/SentencePiece, ROM)    │  │  (top-p/k)  │  │
│  └─────┬──────┘  └──────┬───────┘  └──────┬──────┘  │
│        │                │                 │          │
│  ┌─────▼────────────────▼─────────────────▼──────┐  │
│  │           Inference Engine (INT4 GEMM)         │  │
│  └────────────────────┬───────────────────────────┘  │
│                       │                              │
│  ┌────────────────────▼───────────────────────────┐  │
│  │     Model Weights (ROM / Flash, INT4)          │  │
│  │     Tiny-Transformer 50 M params               │  │
│  └────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### 4.1  Model Architecture – TinyFormer-USB

- **Architecture**: Decoder-only Transformer (GPT-style)
- **Parameters**: 50 M (target) or 10–100 M range
- **Layers**: 12 transformer blocks
- **d_model**: 512, **n_heads**: 8, **d_ff**: 2048
- **Context window**: 512 tokens (fits in on-device SRAM)
- **Vocabulary**: 16 384 tokens (BPE)
- **Quantisation**: INT4 weights + INT8 activations (W4A8)

### 4.2  Tokenizer

- SentencePiece BPE, vocab 16 384
- Lookup table stored in on-device Flash
- Merge rules encoded as sorted array for O(n log n) encoding

### 4.3  KV Cache

- Fixed-size circular buffer: `2 × layers × heads × head_dim × context_len`
- For TinyFormer-USB: `2 × 12 × 8 × 64 × 512 × 1 byte = 6 MB` (INT8 cache)
- Fits comfortably in LPDDR5 on-package RAM

### 4.4  Inference Engine

1. **Embedding lookup** – INT4 → INT8 dequant
2. **Multi-Head Attention** – INT8 GEMM on Q/K/V projections
3. **Attention score** – FP16 softmax (small matrix, affordable)
4. **Feed-Forward Network** – W4A8 GEMM + SiLU activation
5. **RMS Norm** – FP16, fused with dequant
6. **Sampling** – top-k / top-p in FP16

---

## 5. "Hardcoded" Implementation Strategy

"Hardcoding everything" means the device is **self-contained**: host only sends
prompt text and receives token text.  No model files are transferred over USB.

| Component | How it is hardcoded |
|---|---|
| Model weights | Flashed to SPI NOR / eMMC at manufacturing time; read-only |
| Tokenizer vocab | Flashed alongside weights; mapped to ROM addresses |
| Inference firmware | FPGA bitstream or MCU firmware in on-chip Flash |
| Sampling parameters | Default values in firmware; configurable via USB control transfers |
| USB descriptor | Hardcoded VID/PID, string descriptors in firmware |

### 5.1  Memory Map (example for 50 M INT4 model)

```
Address        Size     Content
0x00000000  25 MB    INT4 model weights (50 M params × 0.5 B)
0x01900000   2 MB    INT8 embedding table (16384 × 128 B)
0x01B00000   1 MB    Tokenizer BPE merge table
0x01C00000 512 KB    Firmware (RISC-V or ARM Cortex-M7)
0x01C80000 remaining  Reserved / future use
```

---

## 6. Power Budget (USB-A 3.2 @ 900 mA = 4.5 W)

| Component | Power |
|---|---|
| NPU / FPGA (active inference) | 2.0–3.0 W |
| LPDDR5 RAM | 0.3 W |
| SPI Flash (sequential read) | 0.1 W |
| USB PHY + controller | 0.2 W |
| Regulators (LDO losses) | 0.2 W |
| **Total** | **~2.8–3.8 W** ✓ |

---

## 7. Achievability Summary

| Requirement | Feasibility | Notes |
|---|---|---|
| USB form factor | ✅ Feasible | Custom 2-layer PCB ~60 × 20 mm |
| 20 kt/s throughput | ✅ Feasible with INT4 | Requires ≤ 100 M param model + custom NPU |
| Hardcoded weights | ✅ Standard practice | SPI NOR or eMMC flash at assembly |
| Hardcoded CPU/GPU | ✅ FPGA bitstream or ASIC | One-time programming |
| USB-powered (≤ 4.5 W) | ✅ Feasible | Careful power management required |
| No host-side model | ✅ Self-contained firmware | USB bulk-transfer API only |

---

## 8. References & Further Reading

- Dettmers et al., "LLM.int8()" (2022) – 8-bit quantisation
- Frantar et al., "GPTQ" (2022) – accurate INT4 post-training quantisation
- Ma et al., "Era of 1-bit LLMs" (2024) – BitNet extreme quantisation
- Hailo-8 Datasheet – 26 TOPS NPU in M.2/USB form factor
- llama.cpp project – CPU/GPU optimised inference reference
- GGUF format – packed weight format suitable for on-device storage
