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
