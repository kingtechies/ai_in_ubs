# USB AI Accelerator - Complete Research & Development Plan

## Project EMPIRE: High-Performance Portable AI

**Target: 20,000 tokens/second in USB form factor**

This document consolidates all research for building a USB device with hardcoded AI model, custom GPU/NPU, CPU, and memory - achieving unprecedented inference speeds.

---

# Table of Contents

1. [Executive Summary](#executive-summary)
2. [Technical Specifications](#technical-specifications)
3. [Hardware Architecture](#hardware-architecture)
4. [Software Architecture](#software-architecture)
5. [Performance Optimization](#performance-optimization)
6. [Implementation Roadmap](#implementation-roadmap)
7. [Bill of Materials](#bill-of-materials)
8. [Comparison with Existing Solutions](#comparison-with-existing-solutions)

---

# 1. Executive Summary

## Project Vision

Building a portable USB device that runs AI models at **20,000 tokens per second (20kt/s)** with all components hardcoded into the device - GPU, CPU, memory, and model weights.

```
┌─────────────────────────────────────────────────────────────────┐
│                    USB AI ACCELERATOR                            │
│                                                                 │
│     ┌──┐      ┌────────────────────────────────────────────┐   │
│     │  │══════│  PMU │ NPU (50+ TOPS) │ DRAM │ Flash     │   │
│     │  │══════│      │    Tensor      │ 8GB  │ 64GB      │   │
│     │  │══════│      │    Cores       │PoP   │ Model     │   │
│     └──┘      └────────────────────────────────────────────┘   │
│    USB-C             Aluminum Heat Spreader                     │
│                                                                 │
│  Target: 20,000 tokens/sec │ Power: <15W │ Size: USB stick    │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

- **Custom NPU**: 50-100 TOPS INT8 Neural Processing Unit
- **High-bandwidth Memory**: 8-16 GB LPDDR5X with 100+ GB/s bandwidth
- **Pre-loaded Model**: Quantized 1-3B parameter model (INT4)
- **Optimized Firmware**: Hardcoded inference engine with Flash Attention

## Key Specifications Summary

| Specification | Target |
|--------------|--------|
| **Throughput** | 20,000 tokens/second |
| **Latency** | <50ms time to first token |
| **Power** | <15W (USB PD compatible) |
| **Form Factor** | USB-C stick (80×25×12mm) |
| **Model Size** | 1-3B parameters (INT4) |
| **Context Length** | Up to 8192 tokens |
| **Memory** | 8-16 GB LPDDR5X |
| **Storage** | 64-128 GB NVMe |

## How It Works

```
┌─────────────────┐      USB 3.2/4       ┌────────────────────────┐
│   Host PC       │◄════════════════════►│   USB AI Accelerator   │
│                 │    20 Gbps           │                        │
│  ┌───────────┐  │                      │  ┌──────────────────┐  │
│  │ Python/   │  │  1. Send prompt      │  │ USB Controller   │  │
│  │ C++ API   │──┼─────────────────────►│──│                  │  │
│  └───────────┘  │                      │  └────────┬─────────┘  │
│                 │                      │           │            │
│                 │                      │  ┌────────▼─────────┐  │
│                 │                      │  │ Tokenizer        │  │
│                 │                      │  │ (On-device)      │  │
│                 │                      │  └────────┬─────────┘  │
│                 │                      │           │            │
│                 │  4. Stream tokens    │  ┌────────▼─────────┐  │
│                 │◄─────────────────────│──│ NPU Inference    │  │
│                 │   20kt/s             │  │ (Hardcoded Model)│  │
│                 │                      │  └──────────────────┘  │
└─────────────────┘                      └────────────────────────┘
```

### Key Technologies

1. **INT4 Quantization**: Model weights compressed to 4-bit integers (GPTQ/AWQ)
2. **Flash Attention**: O(N) memory attention with hardware acceleration
3. **Speculative Decoding**: 2-3x speedup using draft model verification
4. **Paged KV Cache**: Efficient memory management for long contexts
5. **Layer Streaming**: Prefetch weights while computing current layer

---

# Technical Specifications for USB AI Accelerator

## Target Performance: 20,000 Tokens/Second (20kt/s)

### Performance Requirements Breakdown

| Metric | Target | Notes |
|--------|--------|-------|
| Throughput | 20,000 tokens/sec | For optimized small models |
| Latency | <50ms first token | Time to first token |
| Power Consumption | <15W | USB 3.2 Gen 2x2 max power |
| Form Factor | USB-A or USB-C stick | 10cm x 3cm x 1.5cm max |

---

## 1. Processing Unit Specifications

### Option A: Custom ASIC NPU

```
┌─────────────────────────────────────────────────────────────┐
│                    Custom NPU Architecture                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Matrix Processing Engines              │    │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐     │    │
│  │  │MPE-0 │ │MPE-1 │ │MPE-2 │ │MPE-3 │ │MPE-N │     │    │
│  │  │256x256│ │256x256│ │256x256│ │256x256│ │256x256│    │    │
│  │  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘     │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │         Vector Processing Units (VPU)               │    │
│  │    Activation Functions | Softmax | LayerNorm       │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              On-Chip SRAM Cache                     │    │
│  │              16MB - 64MB Unified                     │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

#### NPU Specifications:
- **Process Node**: 7nm or 5nm FinFET
- **Matrix Engines**: 8-16 MPEs (Matrix Processing Engines)
- **TOPS**: 50-100 TOPS INT8 / 25-50 TFLOPS FP16
- **On-chip SRAM**: 32-64 MB
- **Memory Bandwidth**: 256 GB/s (on-chip)

### Option B: FPGA-Based Solution

For prototyping and lower-volume production:

| Component | Specification |
|-----------|--------------|
| FPGA Chip | Xilinx Zynq UltraScale+ or Intel Agilex |
| Logic Elements | 500K+ LUTs |
| DSP Blocks | 2000+ |
| On-chip RAM | 30+ MB |
| Processing | Custom systolic array design |

### Option C: GPU-Based (Compact)

For higher power budgets:

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA Jetson Orin NX or Custom |
| CUDA Cores | 1024+ |
| Tensor Cores | 32+ |
| Memory | 8-16 GB LPDDR5 |
| Power | 15-25W |

---

## 2. Memory Architecture

### High-Bandwidth Memory Requirements

For 20kt/s with a typical transformer model:
- **Model Size**: Target 1-7B parameters (quantized)
- **Quantization**: INT4/INT8 for weights, FP16 for activations
- **Memory Required**: 
  - 1B params @ INT4: ~0.5 GB
  - 3B params @ INT4: ~1.5 GB
  - 7B params @ INT4: ~3.5 GB

### Memory Configuration

```
┌─────────────────────────────────────────────────────────────┐
│                   Memory Hierarchy                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   L1 Cache (per MPE)                                        │
│   ┌────────────────────┐                                    │
│   │  64-256 KB SRAM    │  <1 cycle latency                 │
│   └────────────────────┘                                    │
│              ↓                                              │
│   L2 Unified Cache                                          │
│   ┌────────────────────┐                                    │
│   │   16-64 MB SRAM    │  2-5 cycle latency                │
│   └────────────────────┘                                    │
│              ↓                                              │
│   Main Memory (Weight Storage)                              │
│   ┌────────────────────┐                                    │
│   │  4-16 GB LPDDR5X   │  ~100 cycle latency               │
│   │  or HBM2e stacked  │  Bandwidth: 100-256 GB/s          │
│   └────────────────────┘                                    │
│              ↓                                              │
│   Weight Storage (Flash)                                    │
│   ┌────────────────────┐                                    │
│   │  32-128 GB NVMe    │  Persistent model storage         │
│   └────────────────────┘                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Thermal Management

### Power Budget Analysis

| Component | Power (W) | % of Total |
|-----------|-----------|------------|
| NPU/Processing | 8-10W | 60% |
| Memory (LPDDR5X) | 2-3W | 20% |
| Controller/IO | 1-2W | 10% |
| Thermal Management | 1W | 7% |
| Misc | 0.5W | 3% |
| **Total** | **12.5-16.5W** | **100%** |

### Thermal Solution

```
┌─────────────────────────────────────────────────────────────┐
│              USB Stick Form Factor (Cross Section)          │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────┐  │
│  │            Aluminum/Copper Heat Spreader              │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │     Thermal Pad / Graphene Heat Dissipator     │  │  │
│  │  │  ┌───────────────────────────────────────────┐  │  │  │
│  │  │  │              NPU Die                       │  │  │  │
│  │  │  │  ┌───────┐  ┌───────┐  ┌───────┐        │  │  │  │
│  │  │  │  │Memory │  │Memory │  │Flash  │        │  │  │  │
│  │  │  │  │Stack  │  │Stack  │  │NVMe   │        │  │  │  │
│  │  │  │  └───────┘  └───────┘  └───────┘        │  │  │  │
│  │  │  └───────────────────────────────────────────┘  │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
│                           ↑                                 │
│                    USB-C Connector                          │
└─────────────────────────────────────────────────────────────┘
```

- **Active Cooling**: Micro-fan or piezoelectric cooling
- **Passive Cooling**: Aluminum housing as heatsink
- **Thermal Throttling**: Dynamic frequency scaling based on temperature

---

## 4. Interface Specifications

### USB Interface

| Standard | Speed | Power |
|----------|-------|-------|
| USB 3.2 Gen 2x2 | 20 Gbps | 15W (PD) |
| USB4 | 40 Gbps | 100W (PD) |
| Thunderbolt 4 | 40 Gbps | 100W (PD) |

### Communication Protocol

```
Host System ←→ USB Controller ←→ Command Queue ←→ NPU
     ↑                                              ↓
     └────────── Response Buffer ←─────────────────┘
```

- **Command Interface**: Custom DMA-based inference requests
- **Data Format**: Tokenized input → Token output
- **Driver Requirements**: Custom kernel driver or libusb

---

## 5. Quantization Requirements for 20kt/s

### Model Quantization Strategy

| Quantization | Model Size Reduction | Speed Impact | Quality Impact |
|--------------|---------------------|--------------|----------------|
| FP32 → FP16 | 2x | +50% | Minimal |
| FP16 → INT8 | 2x | +100% | Low |
| INT8 → INT4 | 2x | +150% | Moderate |

### Recommended Configuration for 20kt/s

For achieving 20,000 tokens/second:

```python
# Recommended Model Configuration
# 
# Quantization Methods:
# - GPTQ: Post-Training Quantization via Gradient-based optimization
#   (https://arxiv.org/abs/2210.17323) - accurate, widely supported
# - AWQ: Activation-aware Weight Quantization
#   (https://arxiv.org/abs/2306.00978) - better quality, activation-aware

model_config = {
    "architecture": "Transformer (Decoder-only)",
    "parameters": "1-3 Billion",
    "quantization": {
        "weights": "INT4 (GPTQ/AWQ)",  # 4-bit integer weights
        "activations": "INT8/FP16",     # 8-bit or 16-bit activations
        "kv_cache": "INT8"              # 8-bit key-value cache
    },
    "attention": {
        "type": "Grouped Query Attention (GQA)",
        "num_kv_heads": 8,  # Reduced from 32
        "head_dim": 128
    },
    "optimizations": [
        "Flash Attention 2",
        "Speculative Decoding",
        "Continuous Batching",
        "Paged Attention"
    ]
}
```

---

## 6. Bill of Materials (Estimated)

### Prototype/Development

| Component | Estimated Cost |
|-----------|----------------|
| Custom ASIC NRE | $5-50M |
| FPGA Development Kit | $5,000-20,000 |
| Memory (HBM2e) | $200-500 |
| PCB & Assembly | $100-500 |
| Enclosure & Thermal | $50-200 |

### Production Unit (at scale)

| Component | Est. Cost @ 100K units |
|-----------|----------------------|
| Custom NPU | $50-150 |
| Memory (LPDDR5X 8GB) | $30-50 |
| Flash Storage (64GB) | $10-20 |
| PCB & Assembly | $15-30 |
| Enclosure | $5-10 |
| **Total BOM** | **$110-260** |

---

## Next Steps

1. Review [Hardware Architecture](hardware-architecture.md) for detailed chip design
2. See [Software Architecture](software-architecture.md) for firmware implementation
3. Check [Performance Optimization](performance-optimization.md) for achieving 20kt/s

---

# Hardware Architecture for USB AI Accelerator

## System-on-Chip (SoC) Design

The heart of the USB AI Accelerator is a custom SoC designed specifically for transformer inference.

---

## 1. Overall Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        USB AI Accelerator SoC                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Neural Processing Unit (NPU)                    │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │            Tensor Core Array (256 TOPS INT8)                  │   │   │
│  │  │  ┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐          │   │   │
│  │  │  │TC0 ││TC1 ││TC2 ││TC3 ││TC4 ││TC5 ││TC6 ││TC7 │          │   │   │
│  │  │  │16x16│16x16│16x16│16x16│16x16│16x16│16x16│16x16│          │   │   │
│  │  │  └────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘          │   │   │
│  │  │  ┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐          │   │   │
│  │  │  │TC8 ││TC9 ││TC10││TC11││TC12││TC13││TC14││TC15│          │   │   │
│  │  │  └────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘          │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  │                                                                     │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐   │   │
│  │  │ Vector Units    │  │ Activation Unit │  │ Attention Engine │   │   │
│  │  │ (SIMD 512-bit)  │  │ (GELU/SiLU/etc)│  │ (Flash Attn HW)  │   │   │
│  │  └─────────────────┘  └─────────────────┘  └──────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Control & Scheduling Unit                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  ┌────────────┐  │   │
│  │  │ ARM Cortex  │  │ DMA Engine  │  │ Task       │  │ Interrupt  │  │   │
│  │  │ M7 @ 400MHz│  │ 8-Channel   │  │ Scheduler  │  │ Controller │  │   │
│  │  └─────────────┘  └─────────────┘  └────────────┘  └────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Memory Subsystem                                │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │              L2 Unified SRAM (32 MB)                        │    │   │
│  │  │    Weight Buffer | Activation Buffer | KV Cache Buffer      │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  │  ┌───────────────────┐  ┌───────────────────────────────────────┐  │   │
│  │  │ Memory Controller │  │       LPDDR5X Interface              │  │   │
│  │  │ (256-bit wide)    │  │       4x16-bit channels              │  │   │
│  │  └───────────────────┘  └───────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      I/O Subsystem                                   │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  ┌────────────┐  │   │
│  │  │ USB 3.2 Gen2│  │ NVMe/UFS   │  │ Boot ROM   │  │ Security   │  │   │
│  │  │ Controller  │  │ Controller  │  │ (64KB)     │  │ Engine     │  │   │
│  │  └─────────────┘  └─────────────┘  └────────────┘  └────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Tensor Core Design

### Matrix Multiplication Unit

Each Tensor Core performs 16x16 matrix operations per cycle:

```
                    Matrix A (16x16)        Matrix B (16x16)
                    ┌───────────────┐       ┌───────────────┐
                    │               │   x   │               │
                    │   INT4/INT8   │       │   INT4/INT8   │
                    │               │       │               │
                    └───────────────┘       └───────────────┘
                              │                     │
                              └──────────┬──────────┘
                                        ↓
                    ┌─────────────────────────────────────┐
                    │        Systolic Array (16x16)        │
                    │                                      │
                    │   PE PE PE PE PE PE PE PE PE ...    │
                    │   PE PE PE PE PE PE PE PE PE ...    │
                    │   PE PE PE PE PE PE PE PE PE ...    │
                    │   ...                               │
                    │                                      │
                    └─────────────────────────────────────┘
                                        │
                                        ↓
                    ┌───────────────────────────────────┐
                    │      Accumulator (INT32/FP32)     │
                    │           + Bias Addition         │
                    │           + Requantization        │
                    └───────────────────────────────────┘
                                        │
                                        ↓
                            Result Matrix C (16x16)
```

### Specifications per Tensor Core:

| Parameter | Value |
|-----------|-------|
| Matrix Size | 16x16 |
| Data Types | INT4, INT8, FP16, BF16 |
| Operations/Cycle | 512 MACs |
| Clock Speed | 1.5-2.0 GHz |
| Power | ~500mW |

### Total NPU Performance:

- **16 Tensor Cores × 512 MACs × 2 GHz = 16.4 TOPS** (INT8)
- **With sparsity (2:4): 32.8 TOPS**
- **With batching + optimizations: 50-100 effective TOPS**

---

## 3. Memory Architecture Deep Dive

### Weight Storage Strategy (Hardcoded Model)

```
┌────────────────────────────────────────────────────────────────┐
│                    Flash Storage Layout                         │
├────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Boot Sector (1 MB)                                        │  │
│  │ - Firmware                                                │  │
│  │ - Model Metadata                                          │  │
│  │ - Configuration                                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Model Weights - Embedding Layer (256 MB)                  │  │
│  │ - Token Embeddings (vocab_size × hidden_dim)             │  │
│  │ - Position Embeddings (if applicable)                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Model Weights - Transformer Layers (2-4 GB)              │  │
│  │ - QKV Projections                                        │  │
│  │ - Output Projections                                     │  │
│  │ - MLP Weights (Up/Down/Gate)                            │  │
│  │ - LayerNorm Parameters                                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ LM Head & Final Layers (256 MB)                          │  │
│  │ - Final LayerNorm                                        │  │
│  │ - Language Model Head                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### Runtime Memory Usage (LPDDR5X)

```
┌──────────────────────────────────────────────────────────────┐
│              Runtime Memory Layout (8 GB LPDDR5X)            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Active Layer Weights Buffer (1-2 GB)                   │  │
│  │ - Prefetched weights for current + next layers         │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ KV Cache (2-4 GB)                                      │  │
│  │ - Paged allocation for dynamic sequence lengths        │  │
│  │ - Supports up to 8192 token context                    │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Activation Memory (1-2 GB)                             │  │
│  │ - Intermediate computations                            │  │
│  │ - Batch processing buffers                             │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ System & DMA Buffers (512 MB)                          │  │
│  │ - USB transfer buffers                                  │  │
│  │ - Command queues                                        │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. Attention Engine (Hardware Flash Attention)

A dedicated hardware unit for efficient attention computation:

```
┌────────────────────────────────────────────────────────────────┐
│                   Hardware Attention Engine                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Input: Q, K, V matrices (from Tensor Cores)                   │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Step 1: QK^T Computation (Tiled)                         │  │
│  │  - Block size: 64x64 or 128x128                          │  │
│  │  - On-chip accumulation                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Step 2: Online Softmax (Numerically Stable)              │  │
│  │  - Running max computation                               │  │
│  │  - Exponential units (lookup table)                      │  │
│  │  - Running sum normalization                             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Step 3: Attention × V                                    │  │
│  │  - Fused with softmax output                             │  │
│  │  - Direct output to activation buffer                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  Benefits:                                                     │
│  - No materialization of full attention matrix                 │
│  - O(N) memory instead of O(N²)                               │
│  - 3-5x speedup over naive implementation                      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 5. Power Delivery Architecture

### USB Power Delivery (PD) Support

```
┌────────────────────────────────────────────────────────────────┐
│                    Power Management Unit                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  USB-C PD Controller                                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Negotiated Voltages:                                     │  │
│  │  - 5V @ 3A = 15W (Default USB 3.x)                      │  │
│  │  - 9V @ 3A = 27W (PD 3.0)                               │  │
│  │  - 15V @ 3A = 45W (PD 3.0)                              │  │
│  │  - 20V @ 5A = 100W (PD 3.0 Extended)                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Voltage Regulators                                       │  │
│  │  - VDD_NPU: 0.7-0.9V (Core logic)                       │  │
│  │  - VDD_SRAM: 0.85V (On-chip memory)                     │  │
│  │  - VDD_IO: 1.8V (LPDDR5X interface)                     │  │
│  │  - VDD_USB: 3.3V (USB PHY)                              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Dynamic Voltage & Frequency Scaling (DVFS)               │  │
│  │  - Performance Mode: 2.0 GHz @ 0.9V (~15W)              │  │
│  │  - Balanced Mode: 1.5 GHz @ 0.8V (~10W)                 │  │
│  │  - Efficiency Mode: 1.0 GHz @ 0.7V (~5W)                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 6. Physical Design Specifications

### Die Size & Package

| Parameter | Specification |
|-----------|--------------|
| Process Node | 7nm/5nm FinFET |
| Die Size | 50-100 mm² |
| Transistor Count | 5-10 Billion |
| Package | FCBGA 15x15mm |
| Ball Count | 400-600 balls |
| TDP | 10-15W |

### USB Stick PCB Layout

```
┌─────────────────────────────────────────────────────────────────┐
│                PCB Layout (Top View)                             │
│                                                                  │
│  USB-C     ┌─────────────────────────────────────────────────┐  │
│  Conn.     │                                                 │  │
│  ┌──┐      │    ┌───────┐   ┌────────────────┐   ┌──────┐   │  │
│  │  │══════│    │ PMU   │   │    SoC/NPU     │   │ DRAM │   │  │
│  │  │══════│    │       │   │                │   │      │   │  │
│  │  │══════│    └───────┘   │    (stacked)   │   │PoP   │   │  │
│  │  │══════│                │                │   │      │   │  │
│  └──┘      │    ┌───────┐   └────────────────┘   └──────┘   │  │
│            │    │Flash  │                                    │  │
│            │    │NVMe   │          ┌──────────────────┐     │  │
│            │    │64-128 │          │  Thermal Sensor  │     │  │
│            │    │GB     │          └──────────────────┘     │  │
│            │    └───────┘                                    │  │
│            └─────────────────────────────────────────────────┘  │
│                                                                  │
│  Dimensions: 80mm x 25mm x 10mm                                 │
│  Weight: ~50g                                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Block-Level Hardware Components

### Required ICs and Components

| Component | Part Number (Reference) | Quantity | Purpose |
|-----------|------------------------|----------|---------|
| NPU SoC | Custom ASIC | 1 | Main AI processor |
| LPDDR5X | Micron MT62F2G32D4 | 2 | 16GB total RAM |
| NVMe Flash | Samsung PM9C1a 128GB | 1 | Model storage |
| USB-C Controller | Cypress CCG6 | 1 | USB PD + data |
| PMIC | Dialog DA9131 | 1 | Power management |
| Temp Sensor | TI TMP117 | 1 | Thermal monitoring |
| ESD Protection | TI TPD4S012 | 1 | USB ESD |
| Oscillator | SiTime SIT8008 | 1 | Reference clock |

---

## 8. Reference Designs

### Existing Hardware Platforms for Reference

1. **Google Coral USB Accelerator**
   - 4 TOPS INT8 (Edge TPU)
   - ~2W power
   - Good baseline architecture

2. **Intel Neural Compute Stick 2**
   - Intel Movidius Myriad X VPU
   - ~1.5W power
   - USB 3.0 interface

3. **Hailo-8 USB Module**
   - 26 TOPS
   - ~2.5W power
   - Advanced dataflow architecture

4. **Groq LPU (Inference Card)**
   - 750 TOPS (larger form factor)
   - Deterministic execution
   - Good architecture reference

---

## Next Steps

1. See [Software Architecture](software-architecture.md) for firmware details
2. Review [Performance Optimization](performance-optimization.md) for achieving 20kt/s
3. Check [Implementation Guide](implementation-guide.md) for development roadmap

---

# Software & Firmware Architecture

## Overview

This document describes the software stack for the USB AI Accelerator, including firmware, drivers, and host-side libraries to achieve 20,000 tokens/second inference.

---

## 1. Software Stack Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HOST COMPUTER                                 │
├─────────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    User Applications                           │  │
│  │  (Python API, C++ API, REST Server, CLI Tools)                │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                   Runtime Library (libusb_ai)                  │  │
│  │  - Model Management                                            │  │
│  │  - Tokenization                                                │  │
│  │  - Request Queuing                                             │  │
│  │  - Response Processing                                         │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    USB Driver Layer                            │  │
│  │  (libusb / Custom Kernel Driver)                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                               │
                          USB 3.2/4
                               │
┌─────────────────────────────────────────────────────────────────────┐
│                      USB AI ACCELERATOR                              │
├─────────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                 USB Controller Firmware                        │  │
│  │  - USB Protocol Handler                                        │  │
│  │  - DMA Management                                              │  │
│  │  - Command Parser                                              │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                NPU Runtime (RTOS-based)                        │  │
│  │  - Task Scheduler                                              │  │
│  │  - Memory Manager                                              │  │
│  │  - Inference Engine                                            │  │
│  │  - KV Cache Manager                                            │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │           Hardware Abstraction Layer (HAL)                     │  │
│  │  - Tensor Core Control                                         │  │
│  │  - Memory Controller                                           │  │
│  │  - DMA Engine                                                  │  │
│  │  - Thermal Management                                          │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Firmware Architecture

### 2.1 Boot Sequence

```
┌────────────────────────────────────────────────────────────┐
│                    Boot Sequence                            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  1. Power-On Reset                                         │
│     │                                                      │
│     ↓                                                      │
│  2. ROM Bootloader (64KB)                                  │
│     - Initialize clocks                                    │
│     - Configure memory controller                          │
│     - Verify firmware signature                            │
│     │                                                      │
│     ↓                                                      │
│  3. Load Main Firmware from Flash                          │
│     - Load to SRAM                                         │
│     - Initialize RTOS                                      │
│     │                                                      │
│     ↓                                                      │
│  4. Hardware Initialization                                │
│     - NPU initialization                                   │
│     - USB PHY initialization                               │
│     - DMA channel setup                                    │
│     │                                                      │
│     ↓                                                      │
│  5. Model Weight Loading                                   │
│     - Load embedding layer to DRAM                         │
│     - Initialize KV cache pages                            │
│     - Prefetch first N layers                              │
│     │                                                      │
│     ↓                                                      │
│  6. Ready State                                            │
│     - USB enumeration complete                             │
│     - Waiting for inference requests                       │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 2.2 RTOS Task Structure

```c
/* Task Priority and Configuration */

// High Priority Tasks
Task_USB_Handler        // Priority: Highest, Period: Interrupt-driven
Task_DMA_Controller     // Priority: High, Period: Interrupt-driven
Task_Thermal_Monitor    // Priority: High, Period: 100ms

// Medium Priority Tasks  
Task_Inference_Engine   // Priority: Medium-High, Period: Continuous
Task_Memory_Manager     // Priority: Medium, Period: On-demand
Task_Weight_Prefetch    // Priority: Medium, Period: Predictive

// Low Priority Tasks
Task_Statistics         // Priority: Low, Period: 1s
Task_Diagnostics        // Priority: Low, Period: On-demand
```

### 2.3 Inference Engine Core

```c
/* inference_engine.c - Hardcoded Transformer Inference */

#include "npu_hal.h"
#include "memory_manager.h"
#include "kv_cache.h"

// Hardcoded model configuration (compiled into firmware)
#define NUM_LAYERS          32
#define HIDDEN_DIM          4096
#define NUM_HEADS           32
#define HEAD_DIM            128
#define INTERMEDIATE_DIM    11008
#define VOCAB_SIZE          32000
#define MAX_SEQ_LEN         8192

// Weight pointers (mapped to flash/DRAM)
typedef struct {
    int4_t* q_proj;          // [hidden_dim, hidden_dim]
    int4_t* k_proj;          // [hidden_dim, kv_dim]
    int4_t* v_proj;          // [hidden_dim, kv_dim]
    int4_t* o_proj;          // [hidden_dim, hidden_dim]
    int4_t* gate_proj;       // [hidden_dim, intermediate_dim]
    int4_t* up_proj;         // [hidden_dim, intermediate_dim]
    int4_t* down_proj;       // [intermediate_dim, hidden_dim]
    fp16_t* rms_norm_weight; // [hidden_dim]
    fp16_t* scales;          // Quantization scales
} LayerWeights;

typedef struct {
    int4_t* token_embedding; // [vocab_size, hidden_dim]
    LayerWeights layers[NUM_LAYERS];
    fp16_t* final_norm;      // [hidden_dim]
    int4_t* lm_head;         // [hidden_dim, vocab_size]
} ModelWeights;

// Global model weights (hardcoded memory addresses)
static const ModelWeights* g_model = (ModelWeights*)MODEL_WEIGHTS_BASE_ADDR;

// Single token forward pass
int32_t forward_single_token(
    int32_t token_id,
    int32_t position,
    KVCache* kv_cache,
    fp16_t* logits_out
) {
    fp16_t hidden_state[HIDDEN_DIM];
    fp16_t residual[HIDDEN_DIM];
    
    // 1. Token Embedding Lookup (hardcoded)
    npu_embedding_lookup(
        g_model->token_embedding,
        token_id,
        hidden_state,
        HIDDEN_DIM
    );
    
    // 2. Process each transformer layer
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        LayerWeights* w = &g_model->layers[layer];
        
        // Prefetch next layer weights (async)
        if (layer + 1 < NUM_LAYERS) {
            memory_prefetch_layer(layer + 1);
        }
        
        // Save residual
        memcpy(residual, hidden_state, sizeof(hidden_state));
        
        // RMSNorm
        npu_rms_norm(hidden_state, w->rms_norm_weight, HIDDEN_DIM);
        
        // Self-Attention (with hardcoded Flash Attention)
        self_attention_forward(
            hidden_state,
            w->q_proj, w->k_proj, w->v_proj, w->o_proj,
            w->scales,
            kv_cache,
            layer,
            position
        );
        
        // Add residual
        npu_vector_add(hidden_state, residual, HIDDEN_DIM);
        memcpy(residual, hidden_state, sizeof(hidden_state));
        
        // RMSNorm (post-attention)
        npu_rms_norm(hidden_state, w->rms_norm_weight + HIDDEN_DIM, HIDDEN_DIM);
        
        // MLP (SwiGLU)
        mlp_forward(
            hidden_state,
            w->gate_proj, w->up_proj, w->down_proj,
            w->scales,
            HIDDEN_DIM,
            INTERMEDIATE_DIM
        );
        
        // Add residual
        npu_vector_add(hidden_state, residual, HIDDEN_DIM);
    }
    
    // 3. Final LayerNorm
    npu_rms_norm(hidden_state, g_model->final_norm, HIDDEN_DIM);
    
    // 4. LM Head (vocabulary projection)
    npu_matmul_int4(
        hidden_state,
        g_model->lm_head,
        logits_out,
        1, HIDDEN_DIM, VOCAB_SIZE
    );
    
    return 0;
}

// Optimized attention with KV cache
static void self_attention_forward(
    fp16_t* hidden_state,
    int4_t* q_proj, int4_t* k_proj, int4_t* v_proj, int4_t* o_proj,
    fp16_t* scales,
    KVCache* cache,
    int layer,
    int position
) {
    // NUM_KV_HEADS defined in model config (typically 8 for GQA, same as NUM_HEADS for MHA)
    // KV_DIM = NUM_KV_HEADS * HEAD_DIM
    fp16_t q[HIDDEN_DIM], k[HEAD_DIM * NUM_KV_HEADS], v[HEAD_DIM * NUM_KV_HEADS];
    fp16_t attn_out[HIDDEN_DIM];
    
    // Compute Q, K, V projections
    npu_matmul_int4_fp16(hidden_state, q_proj, q, 1, HIDDEN_DIM, HIDDEN_DIM, scales);
    npu_matmul_int4_fp16(hidden_state, k_proj, k, 1, HIDDEN_DIM, KV_DIM, scales);
    npu_matmul_int4_fp16(hidden_state, v_proj, v, 1, HIDDEN_DIM, KV_DIM, scales);
    
    // Apply RoPE (Rotary Position Embedding) - hardcoded sin/cos tables
    apply_rope(q, k, position, HEAD_DIM, NUM_HEADS);
    
    // Update KV cache
    kv_cache_append(cache, layer, position, k, v);
    
    // Flash Attention (hardware-accelerated)
    npu_flash_attention(
        q, 
        kv_cache_get_keys(cache, layer),
        kv_cache_get_values(cache, layer),
        attn_out,
        position + 1,  // sequence length
        NUM_HEADS,
        HEAD_DIM
    );
    
    // Output projection
    npu_matmul_int4_fp16(attn_out, o_proj, hidden_state, 1, HIDDEN_DIM, HIDDEN_DIM, scales);
}
```

---

## 3. Host-Side Software

### 3.1 Python API

```python
"""
usb_ai - Python API for USB AI Accelerator
"""

import usb.core
import struct
import numpy as np
from typing import List, Optional, Iterator
from dataclasses import dataclass

@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1


class USBAIAccelerator:
    """
    Python interface to the USB AI Accelerator.
    
    Example usage:
        device = USBAIAccelerator()
        device.connect()
        
        # Generate text
        response = device.generate("Hello, how are you?")
        print(response)
        
        # Streaming generation
        for token in device.generate_stream("Tell me a story"):
            print(token, end="", flush=True)
    """
    
    VENDOR_ID = 0x1234   # Custom vendor ID
    PRODUCT_ID = 0x5678  # Custom product ID
    
    # Command codes
    CMD_GENERATE = 0x01
    CMD_RESET = 0x02
    CMD_GET_STATUS = 0x03
    CMD_SET_CONFIG = 0x04
    CMD_STREAM_START = 0x05
    CMD_STREAM_READ = 0x06
    
    def __init__(self):
        self.device = None
        self.ep_out = None
        self.ep_in = None
        self._tokenizer = None
        
    def connect(self) -> bool:
        """Connect to the USB AI Accelerator."""
        self.device = usb.core.find(
            idVendor=self.VENDOR_ID,
            idProduct=self.PRODUCT_ID
        )
        
        if self.device is None:
            raise RuntimeError("USB AI Accelerator not found")
        
        # Set configuration
        self.device.set_configuration()
        cfg = self.device.get_active_configuration()
        intf = cfg[(0, 0)]
        
        # Get endpoints
        self.ep_out = usb.util.find_descriptor(
            intf,
            custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT
        )
        self.ep_in = usb.util.find_descriptor(
            intf,
            custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN
        )
        
        # Load tokenizer (stored locally, matching model's tokenizer)
        self._load_tokenizer()
        
        return True
    
    def _load_tokenizer(self):
        """Load the tokenizer matching the hardcoded model."""
        # Use sentencepiece or tiktoken based on model
        from tokenizers import Tokenizer
        self._tokenizer = Tokenizer.from_file("tokenizer.json")
    
    def generate(
        self, 
        prompt: str, 
        config: Optional[GenerationConfig] = None
    ) -> str:
        """Generate text completion (blocking)."""
        if config is None:
            config = GenerationConfig()
        
        # Tokenize input
        input_ids = self._tokenizer.encode(prompt).ids
        
        # Build command packet
        packet = self._build_generate_packet(input_ids, config)
        
        # Send to device
        self.ep_out.write(packet)
        
        # Wait for complete response
        output_ids = []
        while True:
            response = self.ep_in.read(4096, timeout=30000)
            status, tokens = self._parse_response(response)
            output_ids.extend(tokens)
            
            if status == 0x00:  # Generation complete
                break
        
        # Decode and return
        return self._tokenizer.decode(output_ids)
    
    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> Iterator[str]:
        """Stream generated tokens one at a time."""
        if config is None:
            config = GenerationConfig()
        
        input_ids = self._tokenizer.encode(prompt).ids
        packet = self._build_stream_packet(input_ids, config)
        
        # Start streaming
        self.ep_out.write(packet)
        
        prev_text = ""
        while True:
            response = self.ep_in.read(512, timeout=5000)
            status, tokens = self._parse_response(response)
            
            if tokens:
                current_text = self._tokenizer.decode(tokens)
                # Yield only new text (handle partial tokens)
                if len(current_text) > len(prev_text):
                    yield current_text[len(prev_text):]
                    prev_text = current_text
            
            if status == 0x00:  # Complete
                break
    
    def _build_generate_packet(
        self, 
        input_ids: List[int], 
        config: GenerationConfig
    ) -> bytes:
        """Build command packet for generation."""
        header = struct.pack(
            "<BHHHBBHH",
            self.CMD_GENERATE,          # Command
            len(input_ids),             # Input length
            config.max_new_tokens,      # Max output tokens
            int(config.temperature * 100),
            int(config.top_p * 100),
            config.top_k,
            int(config.repetition_penalty * 100),
            0  # Reserved
        )
        
        # Pack token IDs
        tokens = struct.pack(f"<{len(input_ids)}I", *input_ids)
        
        return header + tokens
    
    def _parse_response(self, data: bytes):
        """Parse response packet from device."""
        status = data[0]
        num_tokens = struct.unpack("<H", data[1:3])[0]
        tokens = list(struct.unpack(f"<{num_tokens}I", data[3:3+num_tokens*4]))
        return status, tokens
    
    def get_status(self) -> dict:
        """Get device status including temperature, throughput stats."""
        self.ep_out.write(bytes([self.CMD_GET_STATUS]))
        response = self.ep_in.read(64)
        
        return {
            "temperature_c": response[0],
            "power_mw": struct.unpack("<H", response[1:3])[0],
            "tokens_generated": struct.unpack("<I", response[3:7])[0],
            "tokens_per_second": struct.unpack("<H", response[7:9])[0],
            "memory_used_mb": struct.unpack("<H", response[9:11])[0],
            "uptime_seconds": struct.unpack("<I", response[11:15])[0],
        }
    
    def reset(self):
        """Reset the device and clear KV cache."""
        self.ep_out.write(bytes([self.CMD_RESET]))
        response = self.ep_in.read(4, timeout=5000)
        return response[0] == 0x00
    
    def disconnect(self):
        """Disconnect from device."""
        if self.device:
            usb.util.dispose_resources(self.device)
            self.device = None


# Convenience functions
def generate(prompt: str, **kwargs) -> str:
    """Quick generation function."""
    device = USBAIAccelerator()
    device.connect()
    try:
        return device.generate(prompt, GenerationConfig(**kwargs))
    finally:
        device.disconnect()
```

### 3.2 C++ API Header

```cpp
/**
 * usb_ai.hpp - C++ API for USB AI Accelerator
 */

#ifndef USB_AI_HPP
#define USB_AI_HPP

#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <cstdint>

namespace usb_ai {

struct GenerationConfig {
    uint16_t max_new_tokens = 256;
    float temperature = 0.7f;
    float top_p = 0.9f;
    uint8_t top_k = 50;
    float repetition_penalty = 1.1f;
};

struct DeviceStatus {
    uint8_t temperature_c;
    uint16_t power_mw;
    uint32_t tokens_generated;
    uint16_t tokens_per_second;
    uint16_t memory_used_mb;
    uint32_t uptime_seconds;
};

using StreamCallback = std::function<void(const std::string& token)>;

class USBAIAccelerator {
public:
    USBAIAccelerator();
    ~USBAIAccelerator();
    
    // Connection management
    bool connect();
    void disconnect();
    bool is_connected() const;
    
    // Generation
    std::string generate(
        const std::string& prompt,
        const GenerationConfig& config = GenerationConfig{}
    );
    
    void generate_stream(
        const std::string& prompt,
        StreamCallback callback,
        const GenerationConfig& config = GenerationConfig{}
    );
    
    // Device control
    DeviceStatus get_status();
    bool reset();
    
private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace usb_ai

#endif // USB_AI_HPP
```

---

## 4. Hardcoded Model Integration

### 4.1 Model Compilation Pipeline

The process of "hardcoding" the model into the USB device:

```
┌─────────────────────────────────────────────────────────────────┐
│                 Model Compilation Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Original Model (PyTorch/HuggingFace)                       │
│     └─→ Export to ONNX or custom format                        │
│                                                                 │
│  2. Quantization                                                │
│     └─→ GPTQ/AWQ INT4 quantization                             │
│     └─→ Calibration with representative dataset                │
│                                                                 │
│  3. Weight Packing                                              │
│     └─→ Pack INT4 weights (2 weights per byte)                 │
│     └─→ Generate quantization scales/zeros                     │
│     └─→ Optimize memory layout for NPU                         │
│                                                                 │
│  4. Graph Optimization                                          │
│     └─→ Operator fusion (QKV projection, GeLU, etc.)          │
│     └─→ Memory planning (minimize fragmentation)               │
│     └─→ Generate execution schedule                            │
│                                                                 │
│  5. Code Generation                                             │
│     └─→ Generate firmware inference code                       │
│     └─→ Generate weight loading routines                       │
│     └─→ Compile firmware binary                                │
│                                                                 │
│  6. Flash Image Creation                                        │
│     └─→ Combine: Bootloader + Firmware + Weights               │
│     └─→ Add checksums and signatures                           │
│     └─→ Create flashable image                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Model Compiler Tool

```python
"""
model_compiler.py - Compile PyTorch model for USB AI Accelerator
"""

import torch
import struct
import numpy as np
from pathlib import Path
from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

class ModelCompiler:
    """
    Compiles a transformer model into firmware-compatible format.
    """
    
    def __init__(self, model_name: str, output_dir: Path):
        self.model_name = model_name
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def compile(self, calibration_data: str = None):
        """Full compilation pipeline."""
        print("Step 1: Loading model...")
        model, tokenizer = self._load_model()
        
        print("Step 2: Quantizing to INT4...")
        quantized_model = self._quantize_model(model, tokenizer, calibration_data)
        
        print("Step 3: Extracting and packing weights...")
        packed_weights = self._pack_weights(quantized_model)
        
        print("Step 4: Generating firmware code...")
        self._generate_firmware_code(quantized_model)
        
        print("Step 5: Creating flash image...")
        self._create_flash_image(packed_weights)
        
        print("Compilation complete!")
        
    def _load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        return model, tokenizer
    
    def _quantize_model(self, model, tokenizer, calibration_data):
        """Apply GPTQ quantization."""
        quantize_config = {
            "bits": 4,
            "group_size": 128,
            "desc_act": False,
            "sym": True,
        }
        
        # Use AutoGPTQ for quantization
        model_quantized = AutoGPTQForCausalLM.from_pretrained(
            model,
            quantize_config=quantize_config,
        )
        
        if calibration_data:
            model_quantized.quantize(calibration_data)
        
        return model_quantized
    
    def _pack_weights(self, model) -> Dict[str, bytes]:
        """Pack INT4 weights into binary format."""
        packed = {}
        
        for name, param in model.named_parameters():
            if "weight" in name:
                # Get quantized weights and scales
                weights = param.data.cpu().numpy()
                
                # Pack INT4 (2 values per byte)
                packed_data = self._pack_int4(weights)
                packed[name] = packed_data
                
        return packed
    
    def _pack_int4(self, weights: np.ndarray) -> bytes:
        """Pack INT4 weights into bytes (2 per byte)."""
        flat = weights.flatten().astype(np.int8)
        # Ensure even length
        if len(flat) % 2 != 0:
            flat = np.append(flat, 0)
        
        # Pack pairs into bytes
        packed = ((flat[0::2] & 0x0F) | ((flat[1::2] & 0x0F) << 4))
        return packed.tobytes()
    
    def _generate_firmware_code(self, model):
        """Generate C code for model configuration."""
        config = model.config
        
        code = f"""
/* Auto-generated model configuration */
#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#define MODEL_NAME "{self.model_name}"
#define NUM_LAYERS {config.num_hidden_layers}
#define HIDDEN_DIM {config.hidden_size}
#define NUM_HEADS {config.num_attention_heads}
#define NUM_KV_HEADS {getattr(config, 'num_key_value_heads', config.num_attention_heads)}
#define HEAD_DIM ({config.hidden_size} / {config.num_attention_heads})
#define INTERMEDIATE_DIM {config.intermediate_size}
#define VOCAB_SIZE {config.vocab_size}
#define MAX_SEQ_LEN {getattr(config, 'max_position_embeddings', 4096)}
#define RMS_NORM_EPS {config.rms_norm_eps}f

// Memory layout addresses (set during linking)
#define EMBEDDING_ADDR    0x00000000
#define LAYER_0_ADDR      0x{config.hidden_size * config.vocab_size // 2:08X}

#endif // MODEL_CONFIG_H
"""
        
        with open(self.output_dir / "model_config.h", "w") as f:
            f.write(code)
    
    def _create_flash_image(self, packed_weights: Dict[str, bytes]):
        """Create the final flash image."""
        # Header
        header = struct.pack(
            "<4sIIII",
            b"USBM",  # Magic
            1,        # Version
            sum(len(w) for w in packed_weights.values()),  # Total size
            len(packed_weights),  # Num sections
            0x1000    # Data offset
        )
        
        # Write to file
        with open(self.output_dir / "model.bin", "wb") as f:
            f.write(header)
            f.write(b'\x00' * (0x1000 - len(header)))  # Pad to data offset
            
            for name, data in packed_weights.items():
                f.write(data)
        
        print(f"Flash image created: {self.output_dir / 'model.bin'}")
        print(f"Total size: {sum(len(w) for w in packed_weights.values()) / 1e9:.2f} GB")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--output", default="./compiled_model", help="Output directory")
    args = parser.parse_args()
    
    compiler = ModelCompiler(args.model, Path(args.output))
    compiler.compile()
```

---

## 5. USB Protocol Specification

### 5.1 Command Format

```
┌────────────────────────────────────────────────────────────────┐
│                    USB Command Packet Format                    │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Offset  Size   Field         Description                      │
│  ──────────────────────────────────────────────────────────── │
│  0       1      cmd_type      Command type (0x01-0xFF)        │
│  1       2      payload_len   Length of payload               │
│  3       1      flags         Command flags                   │
│  4       4      sequence_id   Request sequence number         │
│  8       N      payload       Command-specific data           │
│                                                                │
└────────────────────────────────────────────────────────────────┘

Command Types:
  0x01 - GENERATE          Generate tokens (blocking)
  0x02 - STREAM_START      Start streaming generation
  0x03 - STREAM_READ       Read streaming tokens
  0x04 - STREAM_STOP       Stop streaming
  0x05 - RESET             Reset device state
  0x06 - GET_STATUS        Get device status
  0x07 - SET_CONFIG        Set generation config
  0x08 - EMBED             Get embeddings only
  0x09 - LOGITS            Get raw logits
```

### 5.2 Response Format

```
┌────────────────────────────────────────────────────────────────┐
│                    USB Response Packet Format                   │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Offset  Size   Field         Description                      │
│  ──────────────────────────────────────────────────────────── │
│  0       1      status        Status code (0=OK, else error)  │
│  1       2      payload_len   Length of payload               │
│  3       1      flags         Response flags                  │
│  4       4      sequence_id   Matching request ID             │
│  8       N      payload       Response-specific data          │
│                                                                │
└────────────────────────────────────────────────────────────────┘

Status Codes:
  0x00 - SUCCESS           Operation completed
  0x01 - PENDING           More data available (streaming)
  0x10 - ERR_INVALID_CMD   Unknown command
  0x11 - ERR_OVERFLOW      Buffer overflow
  0x12 - ERR_TIMEOUT       Operation timeout
  0x13 - ERR_THERMAL       Thermal throttling
  0x20 - ERR_MODEL         Model inference error
```

---

## 6. KV Cache Management

Efficient KV cache management is critical for achieving 20kt/s:

```c
/* kv_cache.c - Paged KV Cache Implementation */

#include "kv_cache.h"
#include "memory_pool.h"

#define PAGE_SIZE       256   // Tokens per page
#define MAX_PAGES       32    // Max pages per sequence
#define NUM_LAYERS      32

typedef struct {
    fp16_t* key_cache;    // [PAGE_SIZE, num_kv_heads, head_dim]
    fp16_t* value_cache;  // [PAGE_SIZE, num_kv_heads, head_dim]
    uint8_t ref_count;    // For copy-on-write
} KVPage;

typedef struct {
    KVPage* pages[MAX_PAGES];   // Page table
    int num_pages;              // Current pages allocated
    int current_length;         // Current sequence length
} LayerCache;

typedef struct {
    LayerCache layers[NUM_LAYERS];
    MemoryPool* page_pool;
} KVCache;

// Allocate new page from pool
static KVPage* allocate_page(KVCache* cache) {
    KVPage* page = memory_pool_alloc(cache->page_pool, sizeof(KVPage));
    if (page) {
        page->key_cache = memory_pool_alloc(
            cache->page_pool,
            PAGE_SIZE * NUM_KV_HEADS * HEAD_DIM * sizeof(fp16_t)
        );
        page->value_cache = memory_pool_alloc(
            cache->page_pool,
            PAGE_SIZE * NUM_KV_HEADS * HEAD_DIM * sizeof(fp16_t)
        );
        page->ref_count = 1;
    }
    return page;
}

// Append new KV pair to cache
void kv_cache_append(
    KVCache* cache,
    int layer,
    int position,
    fp16_t* key,
    fp16_t* value
) {
    LayerCache* lc = &cache->layers[layer];
    
    int page_idx = position / PAGE_SIZE;
    int page_offset = position % PAGE_SIZE;
    
    // Allocate new page if needed
    while (page_idx >= lc->num_pages) {
        lc->pages[lc->num_pages++] = allocate_page(cache);
    }
    
    KVPage* page = lc->pages[page_idx];
    
    // Copy key and value to page
    size_t kv_size = NUM_KV_HEADS * HEAD_DIM * sizeof(fp16_t);
    memcpy(
        page->key_cache + page_offset * NUM_KV_HEADS * HEAD_DIM,
        key,
        kv_size
    );
    memcpy(
        page->value_cache + page_offset * NUM_KV_HEADS * HEAD_DIM,
        value,
        kv_size
    );
    
    lc->current_length = position + 1;
}

// Get contiguous key tensor for attention
fp16_t* kv_cache_get_keys(KVCache* cache, int layer) {
    // Implementation uses DMA to gather pages into contiguous buffer
    // for efficient attention computation
    return cache->layers[layer].pages[0]->key_cache;
}

// Clear all cache for new sequence
void kv_cache_clear(KVCache* cache) {
    for (int l = 0; l < NUM_LAYERS; l++) {
        LayerCache* lc = &cache->layers[l];
        for (int p = 0; p < lc->num_pages; p++) {
            memory_pool_free(cache->page_pool, lc->pages[p]);
            lc->pages[p] = NULL;
        }
        lc->num_pages = 0;
        lc->current_length = 0;
    }
}
```

---

## Next Steps

1. Review [Performance Optimization](performance-optimization.md) for achieving 20kt/s
2. Check [Implementation Guide](implementation-guide.md) for development roadmap

---

# Performance Optimization for 20,000 Tokens/Second

## Target: 20kt/s Inference Speed

Achieving 20,000 tokens per second requires optimizations at every level of the stack. This document details the strategies and calculations.

---

## 1. Performance Budget Analysis

### Token Generation Breakdown

For autoregressive generation at 20,000 tokens/second:

```
Time per token = 1,000,000 μs / 20,000 tokens = 50 μs per token
```

This 50 μs budget must cover:

| Operation | Typical Time | Optimized Target |
|-----------|--------------|------------------|
| Embedding Lookup | 1 μs | 0.5 μs |
| Attention (per layer) | 20-30 μs | 0.8 μs |
| MLP (per layer) | 15-25 μs | 0.5 μs |
| LayerNorm (per layer) | 2-5 μs | 0.1 μs |
| Output Projection | 5-10 μs | 2 μs |
| USB Transfer Overhead | 10-50 μs | 5 μs |
| **Total (32 layers)** | **~1000+ μs** | **~50 μs** |

---

## 2. Model Selection for 20kt/s

### Achievable Models at 20kt/s

Based on compute requirements, here are realistic model sizes:

| Model Size | Parameters | INT4 Size | Required TOPS | Achievable? |
|------------|------------|-----------|---------------|-------------|
| TinyLlama | 1.1B | ~600 MB | 10-15 TOPS | ✅ Yes |
| Phi-2 | 2.7B | ~1.4 GB | 25-30 TOPS | ✅ Yes |
| LLaMA 3B | 3B | ~1.5 GB | 30-40 TOPS | ✅ Possible |
| Mistral 7B | 7B | ~3.5 GB | 70-80 TOPS | ⚠️ Challenging |
| LLaMA 7B | 7B | ~3.5 GB | 70-80 TOPS | ⚠️ Challenging |
| LLaMA 13B | 13B | ~6.5 GB | 130+ TOPS | ❌ Unlikely |

### Recommended Configuration

For **guaranteed 20kt/s**:
- **Model**: Custom 1-3B parameter model or Phi-2
- **Quantization**: INT4 (GPTQ/AWQ)
- **Architecture**: GQA (Grouped Query Attention) for reduced KV cache
- **Context Length**: Up to 4096 tokens

---

## 3. Hardware Optimization Strategies

### 3.1 Systolic Array Utilization

Maximize matrix multiplication efficiency:

```
┌─────────────────────────────────────────────────────────────┐
│           Systolic Array Utilization Optimization            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Problem: Single token = batch size 1 = poor utilization    │
│                                                             │
│  Solution 1: Speculative Decoding                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Draft Model (small) generates N tokens             │   │
│  │  Target Model verifies in single forward pass       │   │
│  │  Effective batch size = N (typically 4-8)           │   │
│  │  Speedup: 2-3x                                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Solution 2: Continuous Batching                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Multiple concurrent requests batched together      │   │
│  │  New requests can join mid-generation               │   │
│  │  Better utilization with multiple users             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Solution 3: Tensor Parallelism                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Split large matrices across multiple tensor cores  │   │
│  │  Parallel QKV projections                           │   │
│  │  Parallel MLP computations                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Memory Bandwidth Optimization

Memory bandwidth is often the bottleneck:

```
Required Memory Bandwidth Calculation:
─────────────────────────────────────

For a 3B model at 20kt/s:
- Weights accessed per token: ~3 GB (INT4)
- At 20,000 tokens/sec: 3 GB × 20,000 = 60 TB/s (!!)

This is impossible with external memory!

Solution: Weight Reuse & Caching
────────────────────────────────

1. On-chip SRAM caching of hot weights
2. Layer-by-layer streaming
3. Weight prefetching pipeline

Actual memory access pattern:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Flash → DRAM → On-chip SRAM → Tensor Cores               │
│                                                             │
│   Layer N weights loaded to SRAM                           │
│   While Layer N executing:                                  │
│     - Prefetch Layer N+1 weights (async DMA)               │
│   Layer N+1 ready when needed                               │
│                                                             │
│   Effective bandwidth requirement:                          │
│   - Only new tokens' KV cache writes                       │
│   - ~10-50 GB/s achievable with LPDDR5X                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Optimized Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                  Pipelined Inference Data Flow                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Time →  T0    T1    T2    T3    T4    T5    T6    T7         │
│  ────────────────────────────────────────────────────────────  │
│                                                                 │
│  DMA:    [Load L0][Load L1][Load L2][Load L3][Load L4]...     │
│                                                                 │
│  NPU:          [Exec L0][Exec L1][Exec L2][Exec L3]...        │
│                                                                 │
│  USB:                        [Send T1][Send T2][Send T3]...   │
│                                                                 │
│  Legend:                                                        │
│  - DMA loads next layer weights while current executes         │
│  - NPU executes layers in pipeline                             │
│  - USB sends completed tokens while generating next            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Software Optimization Strategies

### 4.1 Flash Attention Implementation

Hardware-accelerated Flash Attention is critical:

```c
/* Optimized Flash Attention for NPU */

void flash_attention_forward(
    fp16_t* Q,           // [num_heads, head_dim]
    fp16_t* K,           // [seq_len, num_kv_heads, head_dim]
    fp16_t* V,           // [seq_len, num_kv_heads, head_dim]
    fp16_t* output,      // [num_heads, head_dim]
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    // Block size for tiling (optimized for on-chip SRAM)
    const int BLOCK_SIZE = 64;
    
    // Scale factor
    const float scale = 1.0f / sqrtf((float)head_dim);
    
    // Per-head processing
    for (int h = 0; h < num_heads; h++) {
        int kv_head = h / (num_heads / num_kv_heads);  // GQA mapping
        
        fp16_t* q = Q + h * head_dim;
        fp16_t running_max = -INFINITY;
        fp16_t running_sum = 0.0f;
        fp16_t acc[head_dim] = {0};
        
        // Process K,V in blocks
        for (int block = 0; block < seq_len; block += BLOCK_SIZE) {
            int block_end = min(block + BLOCK_SIZE, seq_len);
            
            // Load K,V block to SRAM
            fp16_t* k_block = K + block * num_kv_heads * head_dim + kv_head * head_dim;
            fp16_t* v_block = V + block * num_kv_heads * head_dim + kv_head * head_dim;
            
            // Compute attention scores for block: Q @ K^T
            fp16_t scores[BLOCK_SIZE];
            for (int i = 0; i < block_end - block; i++) {
                scores[i] = npu_dot_product(
                    q, 
                    k_block + i * num_kv_heads * head_dim,
                    head_dim
                ) * scale;
            }
            
            // Online softmax update
            fp16_t block_max = npu_max(scores, block_end - block);
            fp16_t old_max = running_max;
            running_max = max(running_max, block_max);
            
            // Rescale previous accumulator
            fp16_t scale_old = expf(old_max - running_max);
            running_sum *= scale_old;
            npu_scale(acc, scale_old, head_dim);
            
            // Add new block contribution
            for (int i = 0; i < block_end - block; i++) {
                fp16_t weight = expf(scores[i] - running_max);
                running_sum += weight;
                npu_axpy(
                    acc, 
                    v_block + i * num_kv_heads * head_dim,
                    weight,
                    head_dim
                );
            }
        }
        
        // Final normalization
        npu_scale(acc, 1.0f / running_sum, head_dim);
        memcpy(output + h * head_dim, acc, head_dim * sizeof(fp16_t));
    }
}
```

### 4.2 Optimized INT4 Matrix Multiplication

```c
/* High-performance INT4 matrix multiplication */

// Dequantize INT4 to FP16 on-the-fly during matmul
void matmul_int4_fp16(
    fp16_t* input,       // [M, K] in FP16
    int4_packed_t* weight, // [K, N] packed INT4
    fp16_t* scales,      // [N / group_size] quantization scales
    fp16_t* output,      // [M, N] in FP16
    int M, int K, int N,
    int group_size
) {
    // Tile sizes for SRAM
    const int TILE_M = 16;
    const int TILE_N = 16;
    const int TILE_K = 64;
    
    // Process in tiles for cache efficiency
    for (int m = 0; m < M; m += TILE_M) {
        for (int n = 0; n < N; n += TILE_N) {
            // Accumulator in FP32 for precision
            float acc[TILE_M][TILE_N] = {{0}};
            
            for (int k = 0; k < K; k += TILE_K) {
                // Load input tile
                fp16_t input_tile[TILE_M][TILE_K];
                load_tile(input, input_tile, m, k, M, K);
                
                // Load and dequantize weight tile
                fp16_t weight_tile[TILE_K][TILE_N];
                dequantize_int4_tile(
                    weight, scales,
                    weight_tile,
                    k, n, K, N,
                    group_size
                );
                
                // Tile matmul using tensor core
                npu_tile_matmul(input_tile, weight_tile, acc);
            }
            
            // Store output tile
            store_tile_fp32_to_fp16(acc, output, m, n, M, N);
        }
    }
}

// Efficient INT4 unpacking
static inline void dequantize_int4_tile(
    int4_packed_t* packed,
    fp16_t* scales,
    fp16_t* out,
    int k_start, int n_start,
    int K, int N,
    int group_size
) {
    for (int k = 0; k < TILE_K; k++) {
        for (int n = 0; n < TILE_N; n += 2) {
            // Each byte contains 2 INT4 values
            int idx = ((k_start + k) * N + n_start + n) / 2;
            uint8_t packed_byte = ((uint8_t*)packed)[idx];
            
            // Unpack
            int4_t val0 = (packed_byte & 0x0F) - 8;  // Signed
            int4_t val1 = ((packed_byte >> 4) & 0x0F) - 8;
            
            // Get scale for this group
            int group_idx = (n_start + n) / group_size;
            fp16_t scale = scales[group_idx];
            
            // Dequantize
            out[k * TILE_N + n] = (fp16_t)val0 * scale;
            out[k * TILE_N + n + 1] = (fp16_t)val1 * scale;
        }
    }
}
```

### 4.3 Speculative Decoding Implementation

```c
/* Speculative decoding for higher throughput */

#define SPECULATION_LENGTH 4  // Draft tokens to generate

typedef struct {
    int32_t tokens[SPECULATION_LENGTH];
    int count;
} DraftSequence;

// Main speculative decoding loop
int speculative_generate(
    int32_t* input_ids,
    int input_len,
    int32_t* output_ids,
    int max_new_tokens,
    KVCache* kv_cache
) {
    int generated = 0;
    
    while (generated < max_new_tokens) {
        // Step 1: Draft model generates speculation
        DraftSequence draft;
        draft_model_generate(
            input_ids, input_len + generated,
            &draft,
            SPECULATION_LENGTH
        );
        
        // Step 2: Target model verifies all draft tokens in parallel
        // This is a BATCHED forward pass (more efficient!)
        fp16_t target_logits[SPECULATION_LENGTH + 1][VOCAB_SIZE];
        target_model_forward_batch(
            input_ids, input_len + generated,
            draft.tokens, draft.count,
            kv_cache,
            target_logits
        );
        
        // Step 3: Accept/reject draft tokens
        int accepted = 0;
        for (int i = 0; i < draft.count; i++) {
            // Sample from target distribution
            int32_t target_token = sample_token(target_logits[i]);
            
            if (target_token == draft.tokens[i]) {
                // Accept draft token
                output_ids[generated++] = draft.tokens[i];
                accepted++;
            } else {
                // Reject and use target's token
                output_ids[generated++] = target_token;
                
                // Rollback KV cache
                kv_cache_truncate(kv_cache, input_len + generated);
                break;
            }
        }
        
        // If all accepted, also take the next target token
        if (accepted == draft.count && generated < max_new_tokens) {
            output_ids[generated++] = sample_token(target_logits[draft.count]);
        }
        
        // Check for EOS
        if (output_ids[generated - 1] == EOS_TOKEN) break;
    }
    
    return generated;
}
```

---

## 5. Benchmarking & Profiling

### 5.1 Performance Metrics

```c
/* Performance monitoring */

typedef struct {
    uint64_t tokens_generated;
    uint64_t total_time_us;
    uint64_t attention_time_us;
    uint64_t mlp_time_us;
    uint64_t memory_time_us;
    uint64_t usb_time_us;
    uint32_t peak_temperature_c;
    uint32_t average_power_mw;
} PerformanceMetrics;

PerformanceMetrics g_metrics;

// Real-time throughput calculation
float get_tokens_per_second(void) {
    if (g_metrics.total_time_us == 0) return 0;
    return (float)g_metrics.tokens_generated * 1000000.0f / 
           (float)g_metrics.total_time_us;
}

// Performance breakdown
void print_performance_report(void) {
    float total_time_s = g_metrics.total_time_us / 1000000.0f;
    
    printf("=== Performance Report ===\n");
    printf("Tokens generated: %llu\n", g_metrics.tokens_generated);
    printf("Total time: %.3f s\n", total_time_s);
    printf("Throughput: %.1f tokens/sec\n", get_tokens_per_second());
    printf("\nBreakdown:\n");
    printf("  Attention: %.1f%%\n", 
           100.0f * g_metrics.attention_time_us / g_metrics.total_time_us);
    printf("  MLP: %.1f%%\n",
           100.0f * g_metrics.mlp_time_us / g_metrics.total_time_us);
    printf("  Memory: %.1f%%\n",
           100.0f * g_metrics.memory_time_us / g_metrics.total_time_us);
    printf("  USB: %.1f%%\n",
           100.0f * g_metrics.usb_time_us / g_metrics.total_time_us);
    printf("\nThermal:\n");
    printf("  Peak temperature: %u°C\n", g_metrics.peak_temperature_c);
    printf("  Average power: %u mW\n", g_metrics.average_power_mw);
}
```

### 5.2 Target Benchmarks

| Benchmark | Target | Unit |
|-----------|--------|------|
| Single Token Latency | <50 | μs |
| Time to First Token | <50 | ms |
| Throughput (batch=1) | 20,000 | tok/s |
| Throughput (batch=8) | 100,000 | tok/s |
| Memory Bandwidth Util | >80% | % |
| Compute Utilization | >70% | % |
| Power Efficiency | 1,500+ | tok/s/W |

---

## 6. Achieving 20kt/s - Summary Checklist

### Hardware Requirements

- [ ] Custom NPU with 50+ TOPS INT8 / 25+ TFLOPS FP16
- [ ] 32+ MB on-chip SRAM for weight caching
- [ ] 8-16 GB LPDDR5X with 100+ GB/s bandwidth
- [ ] Hardware Flash Attention unit
- [ ] Efficient INT4 dequantization unit
- [ ] Multi-channel DMA for weight prefetching

### Software Requirements

- [ ] INT4 GPTQ/AWQ quantized model (1-3B params)
- [ ] Flash Attention 2 implementation
- [ ] Paged KV cache with efficient memory management
- [ ] Speculative decoding (2-3x speedup)
- [ ] Layer-wise weight streaming pipeline
- [ ] Optimized tokenizer (BPE, on-device)

### Model Requirements

- [ ] Grouped Query Attention (GQA) architecture
- [ ] Reduced vocabulary size if possible
- [ ] Optimized attention pattern (sliding window optional)
- [ ] Fine-tuned for INT4 quantization

---

## 7. Comparison with Existing Solutions

| Device | Throughput | Power | Form Factor | Price |
|--------|------------|-------|-------------|-------|
| Our Target | 20,000 tok/s | 15W | USB | ~$200 |
| Google Coral | ~100 tok/s* | 2W | USB | $60 |
| Intel NCS2 | ~50 tok/s* | 1.5W | USB | $70 |
| Hailo-8 | ~500 tok/s* | 2.5W | M.2 | $100 |
| NVIDIA Jetson Orin | ~2,000 tok/s | 25W | Module | $500 |
| Groq LPU | 500+ tok/s | 250W | PCIe | $20K+ |
| Apple M3 Max | ~50 tok/s | 40W | Laptop | $3K+ |

*Estimated for LLM inference

Our USB AI Accelerator aims to achieve **10-100x** better performance than existing USB-form-factor solutions while maintaining similar power consumption.

---

## Next Steps

1. Review [Implementation Guide](implementation-guide.md) for development roadmap
2. Check [Technical Specifications](technical-specifications.md) for component details

---

# Implementation Guide

## Project Roadmap for USB AI Accelerator

This guide provides a phased approach to developing the USB AI Accelerator capable of 20,000 tokens/second inference.

---

## Phase 1: Proof of Concept (3-6 months)

### 1.1 FPGA Prototype

Start with an FPGA-based prototype to validate the architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Phase 1: FPGA Prototype                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Hardware: Xilinx Zynq UltraScale+ or Intel Agilex Dev Kit     │
│                                                                 │
│  Goals:                                                         │
│  ✓ Implement basic tensor core in RTL                          │
│  ✓ Validate INT4/INT8 matmul performance                       │
│  ✓ Test Flash Attention hardware implementation                │
│  ✓ Verify USB 3.0 data transfer rates                          │
│  ✓ Run 1B parameter model at 1,000+ tok/s                      │
│                                                                 │
│  Deliverables:                                                  │
│  - RTL code for tensor core array                              │
│  - Basic firmware and USB driver                               │
│  - Performance benchmarks and bottleneck analysis              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Development Environment Setup

```bash
# Required tools
# 1. FPGA Development
#    - Xilinx Vivado (for Zynq) or Intel Quartus (for Agilex)
#    - Verilog/SystemVerilog for RTL
#    - Verilator for simulation

# 2. Software Development
#    - ARM GCC toolchain (for embedded ARM core)
#    - Python 3.10+ (for model compilation)
#    - PyTorch 2.0+ (for model development)

# 3. USB Development
#    - libusb development headers
#    - USB protocol analyzer (recommended)

# Example: Setting up Python environment
python3 -m venv venv
source venv/bin/activate
pip install torch transformers auto-gptq pyusb numpy
```

### 1.3 Initial RTL Structure

```verilog
// tensor_core.v - Basic tensor core module

module tensor_core #(
    parameter DATA_WIDTH = 4,        // INT4
    parameter ACCUM_WIDTH = 32,      // INT32 accumulator
    parameter MATRIX_SIZE = 16       // 16x16 systolic array
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    
    // Matrix A input (streamed row by row)
    input wire [DATA_WIDTH*MATRIX_SIZE-1:0] a_row,
    input wire a_valid,
    
    // Matrix B input (pre-loaded or streamed)
    input wire [DATA_WIDTH*MATRIX_SIZE-1:0] b_col,
    input wire b_valid,
    
    // Result output
    output reg [ACCUM_WIDTH*MATRIX_SIZE-1:0] c_row,
    output reg c_valid,
    output reg done
);

    // Processing elements (16x16 = 256 PEs)
    reg [ACCUM_WIDTH-1:0] pe_accum [0:MATRIX_SIZE-1][0:MATRIX_SIZE-1];
    reg [DATA_WIDTH-1:0] a_reg [0:MATRIX_SIZE-1][0:MATRIX_SIZE-1];
    reg [DATA_WIDTH-1:0] b_reg [0:MATRIX_SIZE-1][0:MATRIX_SIZE-1];
    
    integer i, j;
    
    // Systolic array computation
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < MATRIX_SIZE; i = i + 1) begin
                for (j = 0; j < MATRIX_SIZE; j = j + 1) begin
                    pe_accum[i][j] <= 0;
                    a_reg[i][j] <= 0;
                    b_reg[i][j] <= 0;
                end
            end
            c_valid <= 0;
            done <= 0;
        end else begin
            // Systolic data flow
            for (i = 0; i < MATRIX_SIZE; i = i + 1) begin
                for (j = 0; j < MATRIX_SIZE; j = j + 1) begin
                    // MAC operation: accumulate += a * b
                    pe_accum[i][j] <= pe_accum[i][j] + 
                        $signed(a_reg[i][j]) * $signed(b_reg[i][j]);
                    
                    // Data movement (systolic flow)
                    if (j > 0) a_reg[i][j] <= a_reg[i][j-1];
                    if (i > 0) b_reg[i][j] <= b_reg[i-1][j];
                end
            end
            
            // Load new A row
            if (a_valid) begin
                for (j = 0; j < MATRIX_SIZE; j = j + 1) begin
                    a_reg[0][j] <= a_row[j*DATA_WIDTH +: DATA_WIDTH];
                end
            end
            
            // Load new B column
            if (b_valid) begin
                for (i = 0; i < MATRIX_SIZE; i = i + 1) begin
                    b_reg[i][0] <= b_col[i*DATA_WIDTH +: DATA_WIDTH];
                end
            end
        end
    end

endmodule
```

---

## Phase 2: Architecture Validation (6-12 months)

### 2.1 Full NPU Implementation on FPGA

```
┌─────────────────────────────────────────────────────────────────┐
│                 Phase 2: Full FPGA Implementation                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Goals:                                                         │
│  ✓ Multiple tensor cores with interconnect                     │
│  ✓ Hardware attention engine                                   │
│  ✓ Memory controller with prefetching                          │
│  ✓ Complete inference pipeline                                 │
│  ✓ Achieve 5,000+ tok/s on FPGA                               │
│                                                                 │
│  Hardware Additions:                                            │
│  - External LPDDR4/5 memory interface                          │
│  - High-speed flash interface (NVMe or UFS)                    │
│  - Thermal monitoring                                          │
│                                                                 │
│  Software Additions:                                            │
│  - Complete firmware RTOS                                       │
│  - Model compiler toolchain                                     │
│  - Host-side Python/C++ API                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Memory Subsystem Implementation

```verilog
// memory_controller.v - Multi-channel memory controller

module memory_controller #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 256,  // 256-bit wide bus
    parameter NUM_CHANNELS = 4
)(
    input wire clk,
    input wire rst_n,
    
    // NPU interface (multiple read ports)
    input wire [NUM_CHANNELS-1:0] read_req,
    input wire [ADDR_WIDTH-1:0] read_addr [0:NUM_CHANNELS-1],
    output reg [DATA_WIDTH-1:0] read_data [0:NUM_CHANNELS-1],
    output reg [NUM_CHANNELS-1:0] read_valid,
    
    // Write interface (for KV cache)
    input wire write_req,
    input wire [ADDR_WIDTH-1:0] write_addr,
    input wire [DATA_WIDTH-1:0] write_data,
    output reg write_done,
    
    // Prefetch interface
    input wire prefetch_req,
    input wire [ADDR_WIDTH-1:0] prefetch_addr,
    input wire [15:0] prefetch_len,  // Words to prefetch
    output reg prefetch_done,
    
    // LPDDR5X PHY interface
    output wire [NUM_CHANNELS-1:0] ddr_ck_p,
    output wire [NUM_CHANNELS-1:0] ddr_ck_n,
    output wire [NUM_CHANNELS-1:0] ddr_cke,
    output wire [ADDR_WIDTH/2-1:0] ddr_addr,
    inout wire [DATA_WIDTH-1:0] ddr_dq,
    // ... additional DDR signals
);

    // Arbitration and scheduling logic
    reg [2:0] state;
    localparam IDLE = 0, READ = 1, WRITE = 2, PREFETCH = 3, REFRESH = 4;
    
    // Prefetch buffer (SRAM)
    reg [DATA_WIDTH-1:0] prefetch_buffer [0:1023];  // 32KB buffer
    reg [9:0] prefetch_head, prefetch_tail;
    
    // Request queue for banking
    // ... implementation details
    
endmodule
```

### 2.3 Host API Development

```python
# host_api/usb_ai_accelerator.py

import ctypes
import usb.core
import usb.util
from typing import List, Generator, Optional
from dataclasses import dataclass
import threading
import queue

@dataclass
class InferenceRequest:
    """Request for inference."""
    tokens: List[int]
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    callback: Optional[callable] = None

class USBAIAccelerator:
    """
    Production-ready API for USB AI Accelerator.
    """
    
    def __init__(self, 
                 vendor_id: int = 0x1234,
                 product_id: int = 0x5678,
                 tokenizer_path: str = "tokenizer.json"):
        self.vendor_id = vendor_id
        self.product_id = product_id
        self.device = None
        self._connected = False
        self._tokenizer = None
        self._load_tokenizer(tokenizer_path)
        
        # Async processing
        self._request_queue = queue.Queue()
        self._response_queue = queue.Queue()
        self._worker_thread = None
        
    def connect(self) -> bool:
        """Establish connection to USB device."""
        self.device = usb.core.find(
            idVendor=self.vendor_id,
            idProduct=self.product_id
        )
        if self.device is None:
            raise ConnectionError("USB AI Accelerator not found")
        
        # Claim interface
        if self.device.is_kernel_driver_active(0):
            self.device.detach_kernel_driver(0)
        self.device.set_configuration()
        
        # Get endpoints
        cfg = self.device.get_active_configuration()
        intf = cfg[(0, 0)]
        
        self._ep_out = usb.util.find_descriptor(
            intf,
            custom_match=lambda e: 
                usb.util.endpoint_direction(e.bEndpointAddress) == 
                usb.util.ENDPOINT_OUT
        )
        self._ep_in = usb.util.find_descriptor(
            intf,
            custom_match=lambda e:
                usb.util.endpoint_direction(e.bEndpointAddress) ==
                usb.util.ENDPOINT_IN
        )
        
        self._connected = True
        self._start_worker()
        return True
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Synchronous text generation."""
        tokens = self._tokenizer.encode(prompt)
        request = InferenceRequest(tokens=tokens, **kwargs)
        
        # Send request
        self._send_request(request)
        
        # Wait for response
        output_tokens = []
        while True:
            response = self._receive_response()
            if response['status'] == 'complete':
                output_tokens.extend(response['tokens'])
                break
            elif response['status'] == 'generating':
                output_tokens.extend(response['tokens'])
        
        return self._tokenizer.decode(output_tokens)
    
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Streaming text generation."""
        tokens = self._tokenizer.encode(prompt)
        request = InferenceRequest(tokens=tokens, **kwargs)
        
        self._send_request(request, streaming=True)
        
        prev_text = ""
        while True:
            response = self._receive_response(timeout=100)
            if response is None:
                continue
            
            if response['tokens']:
                text = self._tokenizer.decode(response['tokens'])
                if len(text) > len(prev_text):
                    yield text[len(prev_text):]
                    prev_text = text
            
            if response['status'] == 'complete':
                break
    
    def get_device_info(self) -> dict:
        """Get device information and stats."""
        self._send_command(0x10)  # GET_INFO command
        info_bytes = self._ep_in.read(256)
        
        return {
            'firmware_version': f"{info_bytes[0]}.{info_bytes[1]}.{info_bytes[2]}",
            'model_name': bytes(info_bytes[3:67]).decode('utf-8').strip('\x00'),
            'model_params_b': struct.unpack('<I', info_bytes[67:71])[0] / 1e9,
            'max_context': struct.unpack('<H', info_bytes[71:73])[0],
            'temperature_c': info_bytes[73],
            'power_mw': struct.unpack('<H', info_bytes[74:76])[0],
            'throughput_tps': struct.unpack('<H', info_bytes[76:78])[0],
        }
    
    def benchmark(self, num_tokens: int = 1000) -> dict:
        """Run performance benchmark."""
        import time
        
        # Warmup
        self.generate("Hello", max_new_tokens=10)
        
        # Benchmark
        start = time.perf_counter()
        output = self.generate(
            "The quick brown fox",
            max_new_tokens=num_tokens,
            temperature=0.0  # Greedy for determinism
        )
        elapsed = time.perf_counter() - start
        
        actual_tokens = len(self._tokenizer.encode(output))
        
        return {
            'tokens_generated': actual_tokens,
            'time_seconds': elapsed,
            'tokens_per_second': actual_tokens / elapsed,
            'ms_per_token': (elapsed * 1000) / actual_tokens
        }
    
    def disconnect(self):
        """Clean disconnect."""
        self._connected = False
        if self._worker_thread:
            self._worker_thread.join(timeout=1.0)
        if self.device:
            usb.util.dispose_resources(self.device)
```

---

## Phase 3: ASIC Development (12-24 months)

### 3.1 ASIC Tape-Out Process

```
┌─────────────────────────────────────────────────────────────────┐
│                    Phase 3: ASIC Development                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Timeline (Total: ~18 months from RTL freeze to tested silicon) │
│  ─────────────────────────────────────────────────────────────  │
│  Month 1-3:  RTL freeze and verification                       │
│  Month 4-6:  Synthesis and place & route                       │
│  Month 7-8:  DRC/LVS and sign-off                              │
│  Month 9-10: Tape-out and mask making                          │
│  Month 11-14: Fabrication                                      │
│  Month 15-16: Packaging                                        │
│  Month 17-18: Testing and characterization                     │
│                                                                 │
│  Key Decisions:                                                 │
│  - Foundry: TSMC, Samsung, or GlobalFoundries                  │
│  - Process: 7nm for cost/performance balance                   │
│  - Package: FCBGA for thermal performance                      │
│                                                                 │
│  Estimated Costs:                                               │
│  - NRE (engineering): $2-5M                                    │
│  - Mask costs (7nm): $5-10M                                    │
│  - First silicon: $500K-1M                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 RTL to GDSII Flow

```tcl
# synthesis.tcl - Example Synopsys Design Compiler script

# Read RTL
read_verilog {
    rtl/tensor_core.v
    rtl/attention_engine.v
    rtl/memory_controller.v
    rtl/npu_top.v
}

# Set design constraints
set_design_top npu_top
read_sdc constraints/timing.sdc

# Technology library
set_target_library "tsmc7nm_sc_typical.db"
set_link_library "* tsmc7nm_sc_typical.db"

# Compile with optimization
compile_ultra -timing_high_effort_script

# Reports
report_timing -significant_digits 4 > reports/timing.rpt
report_area > reports/area.rpt
report_power > reports/power.rpt

# Write output
write -format verilog -output netlist/npu_top.v
write_sdc output/npu_top.sdc
```

---

## Phase 4: Product Development (24-36 months)

### 4.1 USB Stick Form Factor Design

```
┌─────────────────────────────────────────────────────────────────┐
│                  Phase 4: Product Development                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Mechanical Design:                                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │     ┌─────────────────────────────────────────────┐     │   │
│  │     │  ╔═══════════════════════════════════════╗  │     │   │
│  │     │  ║       AI Accelerator PCB             ║  │     │   │
│  │ USB │  ║  ┌───┐ ┌─────────┐ ┌────┐ ┌────┐    ║  │     │   │
│  │  C  │══║══│PMU│ │   NPU   │ │RAM │ │NAND│    ║  │     │   │
│  │     │  ║  └───┘ │ (ASIC)  │ │PoP │ │    │    ║  │     │   │
│  │     │  ║        └─────────┘ └────┘ └────┘    ║  │     │   │
│  │     │  ╚═══════════════════════════════════════╝  │     │   │
│  │     └─────────────────────────────────────────────┘     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Dimensions: 80mm x 25mm x 12mm                                │
│  Weight: <50g                                                   │
│  Material: Aluminum housing (heat dissipation)                 │
│                                                                 │
│  Certifications Required:                                       │
│  - USB-IF certification                                        │
│  - FCC Part 15 (EMC)                                           │
│  - CE marking (Europe)                                         │
│  - RoHS compliance                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Production Test Plan

```python
# production_test.py - Manufacturing test suite

import usb_ai
import time
from dataclasses import dataclass
from typing import List

@dataclass 
class TestResult:
    name: str
    passed: bool
    measured_value: float
    expected_range: tuple
    notes: str = ""

class ProductionTester:
    """Production test suite for USB AI Accelerator."""
    
    def __init__(self, serial_number: str):
        self.serial_number = serial_number
        self.device = None
        self.results: List[TestResult] = []
    
    def run_all_tests(self) -> bool:
        """Run complete test suite."""
        tests = [
            self.test_usb_enumeration,
            self.test_firmware_boot,
            self.test_memory_check,
            self.test_model_load,
            self.test_inference_correctness,
            self.test_throughput,
            self.test_thermal,
            self.test_power_consumption,
            self.test_stress,
        ]
        
        for test in tests:
            try:
                result = test()
                self.results.append(result)
                if not result.passed:
                    print(f"FAIL: {result.name} - {result.notes}")
            except Exception as e:
                self.results.append(TestResult(
                    name=test.__name__,
                    passed=False,
                    measured_value=0,
                    expected_range=(0, 0),
                    notes=str(e)
                ))
        
        return all(r.passed for r in self.results)
    
    def test_usb_enumeration(self) -> TestResult:
        """Test USB device enumeration."""
        start = time.time()
        self.device = usb_ai.USBAIAccelerator()
        connected = self.device.connect()
        elapsed = time.time() - start
        
        return TestResult(
            name="USB Enumeration",
            passed=connected and elapsed < 2.0,
            measured_value=elapsed,
            expected_range=(0, 2.0),
            notes=f"Enumeration time: {elapsed:.3f}s"
        )
    
    def test_throughput(self) -> TestResult:
        """Test inference throughput."""
        benchmark = self.device.benchmark(num_tokens=500)
        tps = benchmark['tokens_per_second']
        
        return TestResult(
            name="Throughput",
            passed=tps >= 18000,  # Allow 10% margin
            measured_value=tps,
            expected_range=(18000, 25000),
            notes=f"Measured: {tps:.0f} tok/s"
        )
    
    def test_thermal(self) -> TestResult:
        """Test thermal performance under load."""
        # Run sustained load
        for _ in range(10):
            self.device.generate("Test " * 100, max_new_tokens=200)
        
        info = self.device.get_device_info()
        temp = info['temperature_c']
        
        return TestResult(
            name="Thermal",
            passed=temp < 85,  # Max safe temp
            measured_value=temp,
            expected_range=(20, 85),
            notes=f"Peak temperature: {temp}°C"
        )
    
    def test_power_consumption(self) -> TestResult:
        """Test power consumption."""
        # Measure during inference
        _ = self.device.generate("Power test", max_new_tokens=100)
        info = self.device.get_device_info()
        power_w = info['power_mw'] / 1000
        
        return TestResult(
            name="Power Consumption",
            passed=power_w <= 15.0,  # USB 3.2 limit
            measured_value=power_w,
            expected_range=(5.0, 15.0),
            notes=f"Power: {power_w:.1f}W"
        )
    
    def generate_report(self) -> str:
        """Generate test report."""
        report = f"Production Test Report - {self.serial_number}\n"
        report += "=" * 60 + "\n"
        
        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            report += f"[{status}] {result.name}: {result.measured_value:.2f} "
            report += f"(expected: {result.expected_range})\n"
            if result.notes:
                report += f"       Notes: {result.notes}\n"
        
        overall = "PASS" if all(r.passed for r in self.results) else "FAIL"
        report += f"\nOverall Result: {overall}\n"
        
        return report


if __name__ == "__main__":
    import sys
    serial = sys.argv[1] if len(sys.argv) > 1 else "TEST001"
    
    tester = ProductionTester(serial)
    passed = tester.run_all_tests()
    print(tester.generate_report())
    
    sys.exit(0 if passed else 1)
```

---

## Phase 5: Scale and Iterate (36+ months)

### 5.1 Future Roadmap

```
┌─────────────────────────────────────────────────────────────────┐
│                      Future Roadmap                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  V1.0 (Initial Release)                                         │
│  ├─ 20,000 tok/s with 3B model                                 │
│  ├─ USB 3.2 Gen 2 interface                                    │
│  └─ 15W power envelope                                          │
│                                                                 │
│  V2.0 (12 months post-launch)                                   │
│  ├─ 50,000 tok/s with improved NPU                             │
│  ├─ Support for 7B models                                       │
│  ├─ USB4/Thunderbolt support                                    │
│  └─ Enhanced multimodal support                                 │
│                                                                 │
│  V3.0 (24 months post-launch)                                   │
│  ├─ 100,000 tok/s                                               │
│  ├─ On-device fine-tuning capability                           │
│  ├─ Multi-device clustering                                     │
│  └─ 5nm process for better efficiency                          │
│                                                                 │
│  Research Directions:                                           │
│  ├─ Mixture-of-Experts hardware support                        │
│  ├─ Sparse attention optimizations                              │
│  ├─ In-memory computing architectures                          │
│  └─ Optical interconnects for bandwidth                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Model Update Mechanism

```python
# model_updater.py - OTA model update system

import hashlib
import struct
from pathlib import Path

class ModelUpdater:
    """
    Secure model update system for USB AI Accelerator.
    """
    
    def __init__(self, device: 'USBAIAccelerator'):
        self.device = device
    
    def check_for_updates(self) -> dict:
        """Check for available model updates."""
        # Query device for current model info
        current = self.device.get_device_info()
        
        # Check update server (simplified)
        # In production, would check signed manifest from server
        return {
            'current_version': current['firmware_version'],
            'available_version': None,  # From server
            'update_available': False,
            'update_size_mb': 0,
        }
    
    def update_model(self, model_path: Path) -> bool:
        """
        Flash new model to device.
        
        The model binary is signed and encrypted for security.
        """
        # Verify model signature
        if not self._verify_signature(model_path):
            raise SecurityError("Model signature verification failed")
        
        # Read model binary
        with open(model_path, 'rb') as f:
            model_data = f.read()
        
        # Calculate checksum
        checksum = hashlib.sha256(model_data).digest()
        
        # Send update command
        self._send_update_header(len(model_data), checksum)
        
        # Stream model data in chunks
        CHUNK_SIZE = 64 * 1024  # 64KB chunks
        for offset in range(0, len(model_data), CHUNK_SIZE):
            chunk = model_data[offset:offset + CHUNK_SIZE]
            self._send_chunk(offset, chunk)
            
            # Progress callback
            progress = (offset + len(chunk)) / len(model_data)
            print(f"Update progress: {progress*100:.1f}%")
        
        # Verify and finalize
        return self._finalize_update()
    
    def _verify_signature(self, model_path: Path) -> bool:
        """Verify model signature using public key."""
        # In production, use proper cryptographic verification
        # (e.g., RSA-2048 or Ed25519)
        return True  # Placeholder
    
    def _send_update_header(self, size: int, checksum: bytes):
        """Send update header to device."""
        header = struct.pack('<BII', 0x50, size, 0) + checksum
        self.device._ep_out.write(header)
    
    def _send_chunk(self, offset: int, data: bytes):
        """Send data chunk to device."""
        header = struct.pack('<BII', 0x51, offset, len(data))
        self.device._ep_out.write(header + data)
        
        # Wait for ACK
        response = self.device._ep_in.read(4)
        if response[0] != 0x00:
            raise IOError(f"Chunk write failed at offset {offset}")
    
    def _finalize_update(self) -> bool:
        """Finalize update and reboot device."""
        self.device._ep_out.write(bytes([0x52]))  # FINALIZE command
        response = self.device._ep_in.read(4, timeout=30000)
        return response[0] == 0x00
```

---

## Resource Links

### Documentation
- [Technical Specifications](technical-specifications.md)
- [Hardware Architecture](hardware-architecture.md)
- [Software Architecture](software-architecture.md)
- [Performance Optimization](performance-optimization.md)

### External Resources
- [LLM Inference Optimization Survey](https://arxiv.org/abs/2303.06865)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [GPTQ Quantization](https://arxiv.org/abs/2210.17323)
- [Speculative Decoding](https://arxiv.org/abs/2211.17192)

### Reference Implementations
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - CPU inference
- [vLLM](https://github.com/vllm-project/vllm) - High-throughput serving
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - NVIDIA optimization


---

# Final Summary & Checklist

## Complete Development Checklist

### Hardware Requirements
- [ ] Custom NPU with 50+ TOPS INT8 / 25+ TFLOPS FP16
- [ ] 32+ MB on-chip SRAM for weight caching
- [ ] 8-16 GB LPDDR5X with 100+ GB/s bandwidth
- [ ] Hardware Flash Attention unit
- [ ] Efficient INT4 dequantization unit
- [ ] Multi-channel DMA for weight prefetching
- [ ] USB-C connector with Power Delivery support
- [ ] Thermal management (aluminum housing, thermal pads)

### Software Requirements
- [ ] INT4 GPTQ/AWQ quantized model (1-3B params)
- [ ] Flash Attention 2 implementation
- [ ] Paged KV cache with efficient memory management
- [ ] Speculative decoding (2-3x speedup)
- [ ] Layer-wise weight streaming pipeline
- [ ] Optimized tokenizer (BPE, on-device)
- [ ] RTOS firmware for real-time inference
- [ ] USB driver (libusb / custom kernel driver)
- [ ] Python and C++ host APIs

### Model Requirements
- [ ] Grouped Query Attention (GQA) architecture
- [ ] Reduced vocabulary size if possible
- [ ] Optimized attention pattern (sliding window optional)
- [ ] Fine-tuned for INT4 quantization

## Timeline Summary

| Phase | Duration | Goal |
|-------|----------|------|
| Phase 1: FPGA Prototype | 3-6 months | 1,000+ tok/s validation |
| Phase 2: Full FPGA | 6-12 months | 5,000+ tok/s |
| Phase 3: ASIC Development | 12-18 months | Tape-out |
| Phase 4: Production | 6-12 months | 20,000+ tok/s |

## Cost Estimates

### Development
- FPGA Development Kit: $5,000-20,000
- ASIC NRE: $5-50M (depending on complexity)
- Software Development: $500K-2M

### Production (at 100K units)
- Per unit BOM: $110-260
- Target retail price: ~$200

---

**Project EMPIRE** - *Empowering AI at the Edge*

*Document Version: 1.0*
*Last Updated: March 17, 2026*
