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
