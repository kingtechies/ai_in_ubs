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
