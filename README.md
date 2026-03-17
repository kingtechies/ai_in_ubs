# AI in UBS – TinyFormer-USB Accelerator

Research and reference implementation for a **USB-form-factor AI inference
device** capable of generating **≥ 20 000 tokens per second (20 kt/s)** with
all model weights, tokenizer, and firmware **hardcoded on the device**.

---

## Goal

Run a small language model entirely self-contained on a USB stick:

- No model files on the host – weights are flashed to on-device SPI NOR / eMMC
- Host sends plain text prompt → device returns generated tokens over USB Bulk
- Target: **20 kt/s** throughput at < 4.5 W (USB bus power)

---

## Architecture

```
Host Computer  ──USB 3.2──►  USB AI Accelerator
                              ├── NPU / FPGA (compute)
                              ├── LPDDR5 / SRAM (KV cache)
                              ├── eMMC Flash (model weights, tokenizer)
                              └── USB controller (firmware)
```

See [`research/USB_AI_ACCELERATOR_RESEARCH.md`](research/USB_AI_ACCELERATOR_RESEARCH.md)
for the full feasibility study, hardware options, and power budget.

---

## Project Structure

```
research/                 Feasibility study & hardware architecture notes
src/
  model/
    tiny_transformer.py   TinyFormer-USB model (decoder-only, 50 M params)
    tokenizer.py          BPE tokenizer (16 384 vocab, on-device serialisable)
  inference/
    engine.py             Single-sequence & async batching inference engine
  hardware/
    usb_interface.py      USB host driver + Flash memory map
    specs.py              Hardware component specs & throughput estimator
benchmarks/
  benchmark_throughput.py Token-throughput & quantisation accuracy benchmark
tests/
  test_usb_ai_accelerator.py  Unit tests for all components
requirements.txt
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Run unit tests
python -m pytest tests/ -v

# Run throughput benchmark (CPU/NumPy baseline)
python benchmarks/benchmark_throughput.py

# Hardware throughput analysis
python src/hardware/specs.py
```

---

## Key Results

| Hardware option | Est. throughput (50 M INT4) | Fits USB-A? | Power |
|---|---|---|---|
| FPGA prototype (Spartan-7) | ~3 kt/s | ✓ | < 2 W |
| **NPU production (Hailo-8L class)** | **~25 kt/s ✓** | **✓** | **< 4 W** |
| GPU tile (M.2 form factor) | ~100 kt/s | ✗ (too large) | ~12 W |

The NPU-based design meets the 20 kt/s target within USB-A bus power limits.

---

## Hardcoded-Everything Strategy

| Component | How it is embedded |
|---|---|
| Model weights | INT4-packed, flashed to eMMC at manufacturing time |
| Tokenizer | BPE merge table in Flash, mapped to ROM addresses |
| Inference firmware | FPGA bitstream or MCU firmware in on-chip Flash |
| Sampling defaults | Hardcoded in firmware; tunable via USB control transfers |
| USB descriptor | VID/PID and device strings in firmware |

---

## References

- [GPTQ: Accurate Post-Training Quantization for GPTs](https://arxiv.org/abs/2210.17323)
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)
- [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)
- [Hailo-8 NPU datasheet](https://hailo.ai/products/hailo-8/)
- [llama.cpp – efficient CPU/GPU inference](https://github.com/ggerganov/llama.cpp)

