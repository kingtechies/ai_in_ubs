"""
USB device hardware abstraction layer.

Defines the communication protocol between the host computer and the
USB AI accelerator device.

Protocol overview
-----------------
The device exposes two USB bulk endpoints:
  EP1 OUT  – host sends prompt (as token IDs or raw text)
  EP1 IN   – device sends generated tokens (streaming)

Control transfers (EP0) are used for:
  - Device information (firmware version, model info, memory map)
  - Sampling parameter configuration
  - Reset / health check

On-device firmware handles:
  - Tokenization (BPE, hardcoded in Flash)
  - Inference (TinyFormer-USB, weights in Flash)
  - Streaming token output back to host

This module provides the HOST-SIDE Python driver to communicate with
the device over libusb.
"""

import struct
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator, Optional


# ---------------------------------------------------------------------------
# USB constants
# ---------------------------------------------------------------------------

VENDOR_ID  = 0x1337    # Reserved for prototype; replace with assigned VID
PRODUCT_ID = 0x0001    # TinyFormer-USB accelerator

# Endpoint addresses
EP_OUT = 0x01          # Bulk OUT (host → device)
EP_IN  = 0x81          # Bulk IN  (device → host)

# Control request types
REQ_TYPE_VENDOR_IN  = 0xC0   # bmRequestType: vendor, device, IN
REQ_TYPE_VENDOR_OUT = 0x40   # bmRequestType: vendor, device, OUT

# Vendor-defined bRequest codes
class UsbRequest(IntEnum):
    GET_DEVICE_INFO  = 0x01
    GET_MODEL_INFO   = 0x02
    SET_TEMPERATURE  = 0x10
    SET_TOP_K        = 0x11
    SET_TOP_P        = 0x12
    SET_MAX_TOKENS   = 0x13
    RESET            = 0x20
    HEALTH_CHECK     = 0x21

# Packet markers in the bulk stream
PACKET_START_PROMPT = 0xA0   # Begin prompt token-ID stream
PACKET_END_PROMPT   = 0xA1   # End of prompt; device starts generating
PACKET_TOKEN        = 0xA2   # One generated token (4 bytes: marker + 3B token_id)
PACKET_END_SEQ      = 0xA3   # End of generated sequence
PACKET_ERROR        = 0xFF   # Device error code follows


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DeviceInfo:
    firmware_version: str
    model_name: str
    vocab_size: int
    n_params: int
    max_seq_len: int
    sram_bytes: int
    flash_bytes: int

    @classmethod
    def from_bytes(cls, data: bytes) -> "DeviceInfo":
        """
        Parse the 64-byte DeviceInfo response from a GET_DEVICE_INFO control
        transfer.

        Binary layout (little-endian):
          [0:4]   firmware version (BCD: major.minor.patch.build)
          [4:36]  model name (32 bytes, null-padded ASCII)
          [36:40] vocab_size (uint32)
          [40:44] n_params   (uint32, in thousands to fit uint32)
          [44:48] max_seq_len (uint32)
          [48:52] sram_bytes  (uint32)
          [52:56] flash_bytes (uint32, in KB)
          [56:64] reserved
        """
        if len(data) < 56:
            raise ValueError(f"DeviceInfo response too short: {len(data)} < 56")
        fw_bcd = struct.unpack_from("<I", data, 0)[0]
        major  = (fw_bcd >> 24) & 0xFF
        minor  = (fw_bcd >> 16) & 0xFF
        patch  = (fw_bcd >>  8) & 0xFF
        build  =  fw_bcd        & 0xFF
        fw_str = f"{major}.{minor}.{patch}.{build}"

        model_raw = data[4:36].rstrip(b"\x00")
        model_name = model_raw.decode("ascii", errors="replace")

        vocab_size, n_params_k, max_seq, sram, flash_kb = struct.unpack_from(
            "<IIIII", data, 36
        )
        return cls(
            firmware_version=fw_str,
            model_name=model_name,
            vocab_size=vocab_size,
            n_params=n_params_k * 1_000,
            max_seq_len=max_seq,
            sram_bytes=sram,
            flash_bytes=flash_kb * 1_024,
        )

    def __str__(self) -> str:
        return (
            f"TinyFormer-USB Device\n"
            f"  Firmware     : {self.firmware_version}\n"
            f"  Model        : {self.model_name}\n"
            f"  Vocabulary   : {self.vocab_size:,} tokens\n"
            f"  Parameters   : {self.n_params / 1e6:.1f} M\n"
            f"  Max context  : {self.max_seq_len} tokens\n"
            f"  SRAM         : {self.sram_bytes / 1024 / 1024:.1f} MB\n"
            f"  Flash        : {self.flash_bytes / 1024 / 1024:.0f} MB\n"
        )


# ---------------------------------------------------------------------------
# Host-side USB driver (requires pyusb / libusb)
# ---------------------------------------------------------------------------

class UsbAIAccelerator:
    """
    Host-side driver for the TinyFormer-USB AI accelerator.

    Usage
    -----
    >>> device = UsbAIAccelerator.find()
    >>> device.open()
    >>> info = device.get_device_info()
    >>> print(info)
    >>> for token_text in device.generate_stream("Hello, world!"):
    ...     print(token_text, end="", flush=True)
    >>> device.close()
    """

    TIMEOUT_MS = 5_000    # default USB transfer timeout

    def __init__(self) -> None:
        self._dev = None    # usb.core.Device (set in open())

    @classmethod
    def find(cls) -> "UsbAIAccelerator":
        """
        Locate the first TinyFormer-USB device connected to the host.
        Raises RuntimeError if no device is found.
        """
        try:
            import usb.core  # type: ignore[import]
            dev = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
            if dev is None:
                raise RuntimeError(
                    f"TinyFormer-USB device not found "
                    f"(VID=0x{VENDOR_ID:04X} PID=0x{PRODUCT_ID:04X}). "
                    "Check USB connection."
                )
            obj = cls()
            obj._dev = dev
            return obj
        except ImportError as exc:
            raise RuntimeError(
                "pyusb is not installed.  Run: pip install pyusb"
            ) from exc

    def open(self) -> None:
        """Claim the USB interface."""
        if self._dev is None:
            raise RuntimeError("Call UsbAIAccelerator.find() first.")
        self._dev.set_configuration()
        self._dev.claim_interface(0)

    def close(self) -> None:
        """Release the USB interface."""
        if self._dev is not None:
            try:
                self._dev.release_interface(0)
            except Exception:
                pass

    def __enter__(self) -> "UsbAIAccelerator":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Control transfers
    # ------------------------------------------------------------------

    def get_device_info(self) -> DeviceInfo:
        """Read device information via control transfer."""
        data = self._dev.ctrl_transfer(
            REQ_TYPE_VENDOR_IN, UsbRequest.GET_DEVICE_INFO,
            wValue=0, wIndex=0, data_or_wLength=64,
            timeout=self.TIMEOUT_MS,
        )
        return DeviceInfo.from_bytes(bytes(data))

    def reset(self) -> None:
        """Reset the device (clears KV-cache and any in-progress generation)."""
        self._dev.ctrl_transfer(
            REQ_TYPE_VENDOR_OUT, UsbRequest.RESET,
            wValue=0, wIndex=0, data_or_wLength=None,
            timeout=self.TIMEOUT_MS,
        )

    def set_temperature(self, value: float) -> None:
        """Set sampling temperature (0.0 = greedy, 1.0 = default)."""
        # Encode as fixed-point Q8.8
        fp = int(value * 256) & 0xFFFF
        self._dev.ctrl_transfer(
            REQ_TYPE_VENDOR_OUT, UsbRequest.SET_TEMPERATURE,
            wValue=fp, wIndex=0, data_or_wLength=None,
            timeout=self.TIMEOUT_MS,
        )

    def set_top_k(self, k: int) -> None:
        """Set top-k sampling parameter (0 = disabled)."""
        self._dev.ctrl_transfer(
            REQ_TYPE_VENDOR_OUT, UsbRequest.SET_TOP_K,
            wValue=k & 0xFFFF, wIndex=0, data_or_wLength=None,
            timeout=self.TIMEOUT_MS,
        )

    def set_max_tokens(self, n: int) -> None:
        """Set maximum number of tokens to generate per request."""
        self._dev.ctrl_transfer(
            REQ_TYPE_VENDOR_OUT, UsbRequest.SET_MAX_TOKENS,
            wValue=n & 0xFFFF, wIndex=0, data_or_wLength=None,
            timeout=self.TIMEOUT_MS,
        )

    # ------------------------------------------------------------------
    # Bulk transfers – prompt submission and token streaming
    # ------------------------------------------------------------------

    def _send_prompt_ids(self, token_ids: list[int]) -> None:
        """
        Send a prompt as a stream of uint16 token IDs over the bulk-out endpoint.

        Packet format:
          [1 B] PACKET_START_PROMPT
          [N × 2 B] token IDs (uint16 little-endian)
          [1 B] PACKET_END_PROMPT
        """
        payload = bytearray()
        payload.append(PACKET_START_PROMPT)
        for tid in token_ids:
            payload.extend(struct.pack("<H", tid & 0xFFFF))
        payload.append(PACKET_END_PROMPT)
        self._dev.write(EP_OUT, payload, timeout=self.TIMEOUT_MS)

    def _receive_token_stream(self) -> Iterator[int]:
        """
        Read generated token IDs from the bulk-in endpoint until EOS or error.

        Yields token IDs (int) as they arrive.
        """
        buf = bytearray()
        while True:
            try:
                chunk = self._dev.read(EP_IN, 512, timeout=self.TIMEOUT_MS)
                buf.extend(chunk)
            except Exception:
                break

            while len(buf) >= 1:
                marker = buf[0]
                if marker == PACKET_TOKEN:
                    if len(buf) < 4:
                        break   # wait for more data
                    token_id = struct.unpack_from("<I", buf, 1)[0] & 0x00FFFFFF
                    buf = buf[4:]
                    yield token_id
                elif marker == PACKET_END_SEQ:
                    buf = buf[1:]
                    return
                elif marker == PACKET_ERROR:
                    error_code = buf[1] if len(buf) > 1 else 0
                    raise RuntimeError(f"Device error code: 0x{error_code:02X}")
                else:
                    # Unknown marker – skip byte and continue
                    buf = buf[1:]

    def generate_stream(
        self,
        prompt_ids: list[int],
    ) -> Iterator[int]:
        """
        Send a prompt and stream back generated token IDs.

        Parameters
        ----------
        prompt_ids : list[int] – pre-tokenised prompt

        Yields
        ------
        token_id : int
        """
        self._send_prompt_ids(prompt_ids)
        yield from self._receive_token_stream()


# ---------------------------------------------------------------------------
# Memory layout specification (on-device Flash)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FlashMemoryMap:
    """
    Address map for the 128 MB SPI NOR Flash on the USB AI accelerator.

    All offsets are byte addresses from the start of the flash chip.
    """
    FIRMWARE_BASE:     int = 0x0000_0000   # 512 KB – MCU/FPGA firmware
    TOKENIZER_BASE:    int = 0x0008_0000   # 1 MB  – BPE tokenizer tables
    WEIGHTS_BASE:      int = 0x0018_0000   # 25 MB – INT4 model weights
    EMBEDDING_BASE:    int = 0x019A_0000   # 2 MB  – INT8 embedding table
    CONFIG_BASE:       int = 0x01BA_0000   # 64 KB – model hyper-parameters
    RESERVED_BASE:     int = 0x01BB_0000   # rest  – future use
    FLASH_SIZE_BYTES:  int = 128 * 1024 * 1024

    def describe(self) -> str:
        lines = ["Flash Memory Map (128 MB SPI NOR):"]
        regions = [
            ("Firmware",   self.FIRMWARE_BASE,   0x0008_0000),
            ("Tokenizer",  self.TOKENIZER_BASE,  0x0010_0000),
            ("Weights",    self.WEIGHTS_BASE,     0x0182_0000),
            ("Embeddings", self.EMBEDDING_BASE,   0x0020_0000),
            ("Config",     self.CONFIG_BASE,      0x0001_0000),
        ]
        for name, base, size_b in regions:
            end = base + size_b - 1
            size_kb = size_b // 1024
            lines.append(
                f"  0x{base:08X}–0x{end:08X}  ({size_kb:>6} KB)  {name}"
            )
        return "\n".join(lines)


FLASH_MAP = FlashMemoryMap()
