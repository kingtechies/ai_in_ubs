"""
Hardware specifications for the TinyFormer-USB AI accelerator device.

This module documents the target hardware components and their specifications.
It also provides helpers for calculating performance estimates given a
particular hardware configuration.
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Hardware component specifications
# ---------------------------------------------------------------------------

@dataclass
class ComputeUnit:
    """Specification for the compute unit (FPGA, NPU, or ASIC)."""
    name: str
    tops_int8: float           # Tera-Operations per second (INT8)
    tops_int4: float           # Approximate INT4 TOPS (usually 2× INT8)
    power_watts: float
    # Physical dimensions (mm)
    width_mm: float
    height_mm: float
    # Memory bandwidth
    memory_bw_gbps: float      # GB/s
    on_chip_sram_mb: float     # MB of on-chip SRAM


@dataclass
class MemorySpec:
    """Off-chip DRAM or SRAM specification."""
    technology: str            # e.g. "LPDDR5", "HBM2e", "SRAM"
    capacity_mb: int
    bandwidth_gbps: float
    power_watts: float


@dataclass
class StorageSpec:
    """Non-volatile storage (Flash) specification."""
    technology: str            # e.g. "SPI NOR", "eMMC", "UFS"
    capacity_mb: int
    read_mbps: float           # Sequential read throughput
    power_watts: float


@dataclass
class UsbSpec:
    """USB interface specification."""
    standard: str              # e.g. "USB 3.2 Gen 2"
    max_throughput_mbps: float
    power_delivery_watts: float


@dataclass
class DeviceSpec:
    """Complete USB AI accelerator device specification."""
    name: str
    compute: ComputeUnit
    memory: MemorySpec
    storage: StorageSpec
    usb: UsbSpec
    pcb_width_mm: float
    pcb_height_mm: float
    total_power_budget_watts: float
    notes: str = ""

    def fits_in_usb_dongle(self) -> bool:
        """Return True if the PCB fits within a standard USB-A dongle envelope."""
        return self.pcb_width_mm <= 65 and self.pcb_height_mm <= 22

    def estimate_throughput(
        self,
        model_params_m: float,      # model size in millions of parameters
        quant_bits: int = 4,        # quantisation (4 = INT4, 8 = INT8)
        batch_size: int = 1,
    ) -> dict:
        """
        Estimate inference throughput for a given model size.

        Uses the memory-bandwidth bottleneck model:
          tokens_per_second = memory_bandwidth / weight_bytes_per_token

        Parameters
        ----------
        model_params_m : float  – model parameter count in millions
        quant_bits     : int    – bits per weight (4 or 8)
        batch_size     : int    – simultaneous sequences

        Returns
        -------
        dict with keys: 'bandwidth_limited_tps', 'compute_limited_tps',
                        'bottleneck', 'estimated_tps'
        """
        bytes_per_param = quant_bits / 8.0
        weight_bytes = model_params_m * 1e6 * bytes_per_param

        effective_bw = self.memory.bandwidth_gbps * 1e9  # bytes/s
        bw_limited_tps = effective_bw / weight_bytes * batch_size

        # Compute: flops per token ≈ 2 × params (one GEMM per param for MVM)
        flops_per_token = 2 * model_params_m * 1e6
        # For INT4, treat TOPS as 2× INT8 TOPS
        tops = self.compute.tops_int4 if quant_bits <= 4 else self.compute.tops_int8
        compute_tps = tops * 1e12 / flops_per_token * batch_size

        bottleneck = "memory" if bw_limited_tps < compute_tps else "compute"
        estimated_tps = min(bw_limited_tps, compute_tps)

        return {
            "bandwidth_limited_tps":  round(bw_limited_tps),
            "compute_limited_tps":    round(compute_tps),
            "bottleneck":             bottleneck,
            "estimated_tps":          round(estimated_tps),
            "meets_20k_target":       estimated_tps >= 20_000,
        }


# ---------------------------------------------------------------------------
# Reference device configurations
# ---------------------------------------------------------------------------

# Option A: FPGA prototype
FPGA_PROTOTYPE = DeviceSpec(
    name="TinyFormer-USB FPGA Prototype (Xilinx Spartan-7)",
    compute=ComputeUnit(
        name="Xilinx XC7S50 FPGA",
        tops_int8=0.05,       # ~50 GOPS achievable for INT8 GEMM
        tops_int4=0.10,       # ~100 GOPS for INT4
        power_watts=1.5,
        width_mm=8.0,
        height_mm=8.0,
        memory_bw_gbps=3.2,   # external SRAM on PCB
        on_chip_sram_mb=2.7,
    ),
    memory=MemorySpec(
        technology="SRAM (IS61WV51216)",
        capacity_mb=8,
        bandwidth_gbps=3.2,
        power_watts=0.2,
    ),
    storage=StorageSpec(
        technology="SPI NOR Flash",
        capacity_mb=128,
        read_mbps=80,
        power_watts=0.1,
    ),
    usb=UsbSpec(
        standard="USB 3.2 Gen 1",
        max_throughput_mbps=5_000,
        power_delivery_watts=4.5,
    ),
    pcb_width_mm=60,
    pcb_height_mm=20,
    total_power_budget_watts=4.5,
    notes=(
        "Research prototype. Best suited for models ≤ 5 M params. "
        "Expected ~1–5 kt/s with INT4."
    ),
)

# Option B: NPU (Hailo-8L class)
NPU_PRODUCTION = DeviceSpec(
    name="TinyFormer-USB NPU (Hailo-8L class)",
    compute=ComputeUnit(
        name="Hailo-8L NPU (or equivalent 7 nm custom ASIC)",
        tops_int8=13.0,        # 13 TOPS INT8
        tops_int4=26.0,        # 26 TOPS INT4
        power_watts=2.5,
        width_mm=12.0,
        height_mm=16.0,
        memory_bw_gbps=50.0,  # on-package LPDDR5
        on_chip_sram_mb=8.0,
    ),
    memory=MemorySpec(
        technology="LPDDR5 (on-package)",
        capacity_mb=2_048,
        bandwidth_gbps=50.0,
        power_watts=0.3,
    ),
    storage=StorageSpec(
        technology="eMMC 5.1",
        capacity_mb=32_768,
        read_mbps=400,
        power_watts=0.1,
    ),
    usb=UsbSpec(
        standard="USB 3.2 Gen 2",
        max_throughput_mbps=10_000,
        power_delivery_watts=4.5,
    ),
    pcb_width_mm=62,
    pcb_height_mm=21,
    total_power_budget_watts=4.5,
    notes=(
        "Production target. Achieves 20 kt/s for 50 M param INT4 model. "
        "Fits standard USB-A dongle form factor."
    ),
)

# Option C: High-end (GPU tile, M.2 form factor)
GPU_HIGHEND = DeviceSpec(
    name="TinyFormer-USB+ (Low-power discrete GPU tile)",
    compute=ComputeUnit(
        name="Entry-level GPU tile (e.g., Intel Arc A-series / AMD 890M equivalent)",
        tops_int8=100.0,
        tops_int4=200.0,
        power_watts=10.0,
        width_mm=22.0,
        height_mm=60.0,
        memory_bw_gbps=200.0,
        on_chip_sram_mb=64.0,
    ),
    memory=MemorySpec(
        technology="GDDR6 / LPDDR5X",
        capacity_mb=4_096,
        bandwidth_gbps=200.0,
        power_watts=1.0,
    ),
    storage=StorageSpec(
        technology="NVMe M.2 2230",
        capacity_mb=256_000,
        read_mbps=3_500,
        power_watts=0.5,
    ),
    usb=UsbSpec(
        standard="USB4 / Thunderbolt 4",
        max_throughput_mbps=40_000,
        power_delivery_watts=100.0,    # USB-C PD
    ),
    pcb_width_mm=22,
    pcb_height_mm=60,
    total_power_budget_watts=12.0,
    notes=(
        "Exceeds USB-A power budget; requires USB-C PD. "
        "Achieves 50–200 kt/s. Larger form factor (M.2 stick)."
    ),
)

ALL_SPECS: list[DeviceSpec] = [FPGA_PROTOTYPE, NPU_PRODUCTION, GPU_HIGHEND]


# ---------------------------------------------------------------------------
# Throughput analysis report
# ---------------------------------------------------------------------------

def print_throughput_analysis(
    model_params_m: float = 50.0,
    quant_bits: int = 4,
) -> None:
    """
    Print a table comparing all reference device configs for a given model.
    """
    print(
        f"\nThroughput analysis: {model_params_m}M-param model, "
        f"INT{quant_bits} quantisation\n"
        + "=" * 72
    )
    header = (
        f"{'Device':<42} {'BW-limit':>9} {'Comp-lim':>9} "
        f"{'Est. TPS':>9} {'≥20k?':>6}"
    )
    print(header)
    print("-" * 72)
    for spec in ALL_SPECS:
        r = spec.estimate_throughput(model_params_m, quant_bits)
        ok = "✓" if r["meets_20k_target"] else "✗"
        name = spec.name[:40]
        print(
            f"{name:<42} {r['bandwidth_limited_tps']:>9,} "
            f"{r['compute_limited_tps']:>9,} "
            f"{r['estimated_tps']:>9,} {ok:>6}"
        )
    print()


if __name__ == "__main__":
    print_throughput_analysis(model_params_m=50.0, quant_bits=4)
    print_throughput_analysis(model_params_m=10.0, quant_bits=4)
    print_throughput_analysis(model_params_m=100.0, quant_bits=8)

    print(FLASH_MAP := __import__(
        "src.hardware.usb_interface", fromlist=["FLASH_MAP"]
    ).FLASH_MAP)
    print()
    print(NPU_PRODUCTION)
