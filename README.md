# AI in USB - High-Performance Portable AI Accelerator

## Project Vision: EMPIRE

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

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Key Specifications](#key-specifications)
3. [How It Works](#how-it-works)
4. [Documentation](#documentation)
5. [Getting Started](#getting-started)
6. [Roadmap](#roadmap)

---

## Executive Summary

This project aims to create a **USB AI Accelerator** - a self-contained AI inference device in a USB form factor that achieves 20,000 tokens per second throughput. The device will contain:

- **Custom NPU**: 50-100 TOPS INT8 Neural Processing Unit
- **High-bandwidth Memory**: 8-16 GB LPDDR5X with 100+ GB/s bandwidth
- **Pre-loaded Model**: Quantized 1-3B parameter model (INT4)
- **Optimized Firmware**: Hardcoded inference engine with Flash Attention

## Key Specifications

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

## Documentation

| Document | Description |
|----------|-------------|
| 📋 [Technical Specifications](docs/technical-specifications.md) | Detailed hardware specs, memory requirements, thermal design |
| 🔧 [Hardware Architecture](docs/hardware-architecture.md) | SoC design, tensor cores, memory subsystem, power delivery |
| 💻 [Software Architecture](docs/software-architecture.md) | Firmware, RTOS, APIs, model compilation pipeline |
| ⚡ [Performance Optimization](docs/performance-optimization.md) | Achieving 20kt/s, benchmarks, optimization strategies |
| 🚀 [Implementation Guide](docs/implementation-guide.md) | Development roadmap, FPGA prototyping, ASIC development |

## Getting Started

### For Users (Future)

```python
# Install the Python package
pip install usb-ai-accelerator

# Connect and generate
from usb_ai import USBAIAccelerator

device = USBAIAccelerator()
device.connect()

# Generate text
response = device.generate("Explain quantum computing in simple terms")
print(response)

# Streaming generation
for token in device.generate_stream("Write a short story about"):
    print(token, end="", flush=True)

# Check device stats
print(f"Throughput: {device.get_status()['tokens_per_second']} tok/s")
```

### For Developers

1. **Phase 1 (FPGA Prototype)**
   - Set up Xilinx/Intel FPGA development environment
   - Implement tensor core in RTL
   - Validate with 1B model at 1,000+ tok/s

2. **Phase 2 (Full Implementation)**
   - Complete NPU architecture
   - Develop firmware and host APIs
   - Target 5,000+ tok/s on FPGA

3. **Phase 3+ (ASIC)**
   - Tape-out custom silicon
   - Achieve 20,000+ tok/s target

See [Implementation Guide](docs/implementation-guide.md) for detailed roadmap.

## Roadmap

```
┌────────────────────────────────────────────────────────────────┐
│                        Project Roadmap                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Q1-Q2 2024: Research & Design                                 │
│  ├── Architecture specification                                │
│  ├── FPGA development kit selection                           │
│  └── Initial RTL design                                        │
│                                                                │
│  Q3-Q4 2024: FPGA Prototype                                    │
│  ├── Tensor core implementation                                │
│  ├── Memory subsystem                                          │
│  └── Basic inference at 1,000 tok/s                           │
│                                                                │
│  2025: Full FPGA Implementation                                │
│  ├── Flash Attention hardware                                  │
│  ├── Complete firmware & APIs                                  │
│  └── Target 5,000+ tok/s                                       │
│                                                                │
│  2026: ASIC Development                                        │
│  ├── RTL freeze & verification                                 │
│  ├── Tape-out (7nm)                                           │
│  └── First silicon validation                                  │
│                                                                │
│  2027: Product Launch                                          │
│  ├── 20,000 tok/s achieved                                     │
│  ├── USB-IF certification                                      │
│  └── Mass production                                           │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## Comparison with Existing Solutions

| Device | Throughput | Power | Form Factor | Est. Price |
|--------|------------|-------|-------------|------------|
| **USB AI Accelerator (Ours)** | **20,000 tok/s** | **15W** | **USB** | **~$200** |
| Google Coral USB | ~100 tok/s | 2W | USB | $60 |
| Intel NCS2 | ~50 tok/s | 1.5W | USB | $70 |
| Hailo-8 | ~500 tok/s | 2.5W | M.2 | $100 |
| NVIDIA Jetson Orin | ~2,000 tok/s | 25W | Module | $500 |
| Apple M3 Max | ~50 tok/s | 40W | Laptop | $3K+ |

## Contributing

This is an open research project. Contributions are welcome in:

- Hardware design (RTL, verification)
- Firmware development (RTOS, inference engine)
- Software (APIs, model compilation)
- Documentation and research

## License

MIT License - See [LICENSE](LICENSE) for details.

---

**Project EMPIRE** - *Empowering AI at the Edge*
