"""
Unit tests for the TinyFormer-USB core components.

Run with:
    python -m pytest tests/ -v
"""

import struct

import numpy as np
import pytest

from src.model.tiny_transformer import (
    TinyFormerConfig,
    TinyFormerInference,
    dequantize_int4,
    quantize_int4,
    random_weights,
    rms_norm,
    serialize_weights_int4,
    silu,
    softmax,
    MAGIC,
    HEADER_FMT,
)
from src.model.tokenizer import (
    BPETokenizer,
    BOS_ID,
    EOS_ID,
    build_demo_tokenizer,
    MAX_VOCAB_SIZE,
)
from src.hardware.specs import (
    NPU_PRODUCTION,
    FPGA_PROTOTYPE,
    GPU_HIGHEND,
    ALL_SPECS,
)
from src.hardware.usb_interface import (
    DeviceInfo,
    FlashMemoryMap,
    FLASH_MAP,
    VENDOR_ID,
    PRODUCT_ID,
)


# ===========================================================================
# TinyFormerConfig
# ===========================================================================

class TestTinyFormerConfig:
    def test_default_config(self):
        cfg = TinyFormerConfig()
        assert cfg.vocab_size == 16_384
        assert cfg.d_model == 512
        assert cfg.n_heads == 8
        assert cfg.n_layers == 12
        assert cfg.d_ff == 2_048
        assert cfg.max_seq_len == 512

    def test_head_dim_derived(self):
        cfg = TinyFormerConfig(d_model=512, n_heads=8)
        assert cfg.head_dim == 64

    def test_invalid_config_raises(self):
        with pytest.raises(AssertionError):
            TinyFormerConfig(d_model=512, n_heads=7)   # 512 not divisible by 7

    def test_n_params_positive(self):
        cfg = TinyFormerConfig()
        assert cfg.n_params > 0

    def test_weight_sizes(self):
        cfg = TinyFormerConfig()
        sizes = cfg.weight_size_bytes
        assert sizes["fp32"] > sizes["fp16"] > sizes["int8"] > sizes["int4"] > sizes["int2"]

    def test_small_config(self):
        cfg = TinyFormerConfig(vocab_size=256, d_model=64, n_heads=4, n_layers=2, d_ff=256)
        assert cfg.head_dim == 16
        assert cfg.n_params > 0


# ===========================================================================
# INT4 Quantisation
# ===========================================================================

class TestInt4Quantization:
    def test_roundtrip_small_matrix(self):
        rng = np.random.default_rng(0)
        w = rng.standard_normal((8, 16)).astype(np.float32)
        packed, scale, zero = quantize_int4(w)
        w_hat = dequantize_int4(packed, scale, zero)
        assert w_hat.shape == w.shape
        rms_err = float(np.sqrt(np.mean((w - w_hat) ** 2)))
        assert rms_err < 0.3, f"RMS error too large: {rms_err}"

    def test_packed_shape(self):
        w = np.ones((4, 32), dtype=np.float32)
        packed, scale, zero = quantize_int4(w)
        assert packed.shape == (4, 16), f"Expected (4,16), got {packed.shape}"
        assert scale.shape == (4,)

    def test_zero_weights(self):
        w = np.zeros((4, 8), dtype=np.float32)
        packed, scale, zero = quantize_int4(w)
        w_hat = dequantize_int4(packed, scale, zero)
        assert np.allclose(w_hat, 0.0, atol=0.1)

    def test_scale_positive(self):
        rng = np.random.default_rng(1)
        w = rng.standard_normal((16, 32)).astype(np.float32)
        _, scale, _ = quantize_int4(w)
        assert np.all(scale > 0)

    def test_compression_ratio(self):
        w = np.ones((512, 512), dtype=np.float32)
        packed, scale, _ = quantize_int4(w)
        compressed = packed.nbytes + scale.nbytes
        original   = w.nbytes
        ratio = original / compressed
        assert ratio > 3.0, f"Expected >3× compression, got {ratio:.2f}×"


# ===========================================================================
# Activation functions
# ===========================================================================

class TestActivations:
    def test_softmax_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = softmax(x)
        assert abs(result.sum() - 1.0) < 1e-6

    def test_softmax_monotone(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        s = softmax(x)
        assert s[0] < s[1] < s[2]

    def test_silu_zero(self):
        assert abs(float(silu(np.float32(0.0)))) < 1e-7

    def test_silu_positive_for_large_x(self):
        assert float(silu(np.float32(10.0))) > 9.0

    def test_rms_norm_shape(self):
        x = np.ones((3, 64), dtype=np.float32)
        w = np.ones(64, dtype=np.float32)
        out = rms_norm(x, w)
        assert out.shape == (3, 64)

    def test_rms_norm_unit_weight(self):
        x = np.full((1, 4), 2.0, dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        out = rms_norm(x, w)
        assert out.shape == (1, 4)
        assert np.allclose(out, 1.0, atol=1e-5)


# ===========================================================================
# TinyFormerInference
# ===========================================================================

class TestTinyFormerInference:
    @pytest.fixture
    def tiny_config(self):
        return TinyFormerConfig(
            vocab_size=256,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=256,
            max_seq_len=32,
        )

    @pytest.fixture
    def tiny_model(self, tiny_config):
        weights = random_weights(tiny_config, seed=0)
        return TinyFormerInference(tiny_config, weights)

    def test_forward_one_token_shape(self, tiny_model, tiny_config):
        logits = tiny_model.forward_one_token(0)
        assert logits.shape == (tiny_config.vocab_size,)

    def test_forward_multiple_tokens(self, tiny_model, tiny_config):
        for i in range(5):
            logits = tiny_model.forward_one_token(i % tiny_config.vocab_size)
        assert logits.shape == (tiny_config.vocab_size,)

    def test_seq_len_increments(self, tiny_model):
        assert tiny_model.seq_len == 0
        tiny_model.forward_one_token(0)
        assert tiny_model.seq_len == 1
        tiny_model.forward_one_token(1)
        assert tiny_model.seq_len == 2

    def test_reset_clears_cache(self, tiny_model):
        tiny_model.forward_one_token(0)
        assert tiny_model.seq_len == 1
        tiny_model.reset()
        assert tiny_model.seq_len == 0

    def test_context_overflow_raises(self, tiny_model, tiny_config):
        for i in range(tiny_config.max_seq_len):
            tiny_model.forward_one_token(i % tiny_config.vocab_size)
        with pytest.raises(RuntimeError, match="Context window exceeded"):
            tiny_model.forward_one_token(0)

    def test_logits_are_finite(self, tiny_model):
        logits = tiny_model.forward_one_token(0)
        assert np.all(np.isfinite(logits))

    def test_deterministic(self, tiny_config):
        weights = random_weights(tiny_config, seed=42)
        m1 = TinyFormerInference(tiny_config, weights)
        m2 = TinyFormerInference(tiny_config, weights)
        logits1 = m1.forward_one_token(5)
        logits2 = m2.forward_one_token(5)
        assert np.allclose(logits1, logits2)


# ===========================================================================
# Weight serialisation
# ===========================================================================

class TestWeightSerialization:
    def test_header_magic(self):
        cfg = TinyFormerConfig(
            vocab_size=256, d_model=64, n_heads=4, n_layers=2, d_ff=256
        )
        weights = random_weights(cfg)
        data = serialize_weights_int4(cfg, weights)
        assert data[:8] == MAGIC

    def test_header_fields(self):
        cfg = TinyFormerConfig(
            vocab_size=256, d_model=64, n_heads=4, n_layers=2, d_ff=256
        )
        weights = random_weights(cfg)
        data = serialize_weights_int4(cfg, weights)
        _, n_layers, vocab_size = struct.unpack_from(HEADER_FMT, data, 0)
        assert n_layers == cfg.n_layers
        assert vocab_size == cfg.vocab_size

    def test_output_is_bytes(self):
        cfg = TinyFormerConfig(
            vocab_size=256, d_model=64, n_heads=4, n_layers=2, d_ff=256
        )
        weights = random_weights(cfg)
        data = serialize_weights_int4(cfg, weights)
        assert isinstance(data, bytes)
        assert len(data) > 0


# ===========================================================================
# Tokenizer
# ===========================================================================

class TestTokenizer:
    @pytest.fixture
    def tokenizer(self):
        return build_demo_tokenizer()

    def test_encode_decode_roundtrip(self, tokenizer):
        text = "hello"
        ids = tokenizer.encode(text, add_bos=False)
        decoded = tokenizer.decode(ids)
        assert "hello" in decoded

    def test_bos_prepended(self, tokenizer):
        ids = tokenizer.encode("abc", add_bos=True)
        assert ids[0] == BOS_ID

    def test_eos_appended(self, tokenizer):
        ids = tokenizer.encode("abc", add_eos=True)
        assert ids[-1] == EOS_ID

    def test_empty_string(self, tokenizer):
        ids = tokenizer.encode("", add_bos=False, add_eos=False)
        assert isinstance(ids, list)

    def test_serialize_deserialize(self, tokenizer):
        data = tokenizer.to_bytes()
        assert len(data) > 0
        tok2 = BPETokenizer.from_bytes(data)
        assert tok2.vocab == tokenizer.vocab

    def test_bad_magic_raises(self):
        with pytest.raises(ValueError, match="magic"):
            BPETokenizer.from_bytes(b"\x00" * 64)

    def test_vocab_size_limit(self):
        vocab = [str(i) for i in range(MAX_VOCAB_SIZE + 1)]
        with pytest.raises(ValueError, match="too large"):
            BPETokenizer(vocab=vocab, merges=[])


# ===========================================================================
# Hardware Specifications
# ===========================================================================

class TestHardwareSpecs:
    def test_npu_meets_20k_target(self):
        # NPU has 50 GB/s BW → can do 20k TPS for up to 5M INT4 param model
        result = NPU_PRODUCTION.estimate_throughput(
            model_params_m=2.5, quant_bits=4
        )
        assert result["meets_20k_target"], (
            f"NPU expected to meet 20k target for 2.5M model, got {result['estimated_tps']} TPS"
        )

    def test_fpga_does_not_meet_20k_for_50m(self):
        result = FPGA_PROTOTYPE.estimate_throughput(
            model_params_m=50.0, quant_bits=4
        )
        assert not result["meets_20k_target"]

    def test_fpga_meets_20k_for_tiny_model(self):
        # FPGA has 3.2 GB/s BW → can do 20k TPS for up to 0.32M INT4 param model
        result = FPGA_PROTOTYPE.estimate_throughput(
            model_params_m=0.1, quant_bits=4
        )
        assert result["meets_20k_target"], (
            f"FPGA expected to meet 20k for 0.1M model, got {result['estimated_tps']}"
        )

    def test_gpu_highend_exceeds_20k(self):
        # GPU has 200 GB/s BW → can do 20k TPS for up to 20M INT4 param model
        result = GPU_HIGHEND.estimate_throughput(
            model_params_m=10.0, quant_bits=4
        )
        assert result["meets_20k_target"]

    def test_npu_fits_usb_dongle(self):
        assert NPU_PRODUCTION.fits_in_usb_dongle()

    def test_gpu_does_not_fit_usb_dongle(self):
        assert not GPU_HIGHEND.fits_in_usb_dongle()

    def test_estimate_throughput_has_required_keys(self):
        result = NPU_PRODUCTION.estimate_throughput(50.0)
        for key in ("bandwidth_limited_tps", "compute_limited_tps",
                    "bottleneck", "estimated_tps", "meets_20k_target"):
            assert key in result, f"Missing key: {key}"

    def test_all_specs_list(self):
        assert len(ALL_SPECS) == 3


# ===========================================================================
# USB Interface / Flash Memory Map
# ===========================================================================

class TestUsbInterface:
    def test_flash_map_firmware_at_zero(self):
        assert FLASH_MAP.FIRMWARE_BASE == 0

    def test_flash_map_weights_after_tokenizer(self):
        assert FLASH_MAP.WEIGHTS_BASE > FLASH_MAP.TOKENIZER_BASE

    def test_flash_map_describe(self):
        desc = FLASH_MAP.describe()
        assert "Firmware" in desc
        assert "Weights" in desc
        assert "Tokenizer" in desc

    def test_vendor_product_ids(self):
        assert VENDOR_ID == 0x1337
        assert PRODUCT_ID == 0x0001

    def test_device_info_parse(self):
        fw_bcd = (1 << 24) | (0 << 16) | (0 << 8) | 0
        # Must be exactly 32 bytes: 14-char name + 1 null + 17 padding = 32
        model_bytes = b"TinyFormer-USB\x00" + b"\x00" * 17
        vocab_size  = 16_384
        n_params_k  = 50_000
        max_seq     = 512
        sram        = 8 * 1024 * 1024
        flash_kb    = 128 * 1024

        payload  = struct.pack("<I", fw_bcd)
        payload += model_bytes
        payload += struct.pack("<IIIII", vocab_size, n_params_k, max_seq, sram, flash_kb)
        payload += b"\x00" * 8

        info = DeviceInfo.from_bytes(payload)
        assert info.firmware_version == "1.0.0.0"
        assert "TinyFormer-USB" in info.model_name
        assert info.vocab_size == 16_384
        assert info.n_params == 50_000 * 1_000
        assert info.max_seq_len == 512

    def test_device_info_str(self):
        fw_bcd = (2 << 24)
        model_bytes = b"TestModel\x00" + b"\x00" * 22
        payload  = struct.pack("<I", fw_bcd)
        payload += model_bytes
        payload += struct.pack("<IIIII", 256, 10, 128, 1024, 1024)
        payload += b"\x00" * 8
        info = DeviceInfo.from_bytes(payload)
        s = str(info)
        assert "TestModel" in s
        assert "Firmware" in s

    def test_device_info_short_payload_raises(self):
        with pytest.raises(ValueError, match="too short"):
            DeviceInfo.from_bytes(b"\x00" * 10)
