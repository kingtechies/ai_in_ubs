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
