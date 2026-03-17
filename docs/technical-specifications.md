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
