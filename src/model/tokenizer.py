"""
Efficient BPE tokenizer for on-device deployment.

Designed to be compiled to a static lookup table that fits in
the SPI NOR Flash of the USB AI accelerator.

The tokenizer is intentionally minimal:
  - vocabulary size: 16 384 tokens (14-bit token IDs)
  - merge table stored as a sorted array for binary-search lookup
  - no external dependencies beyond the standard library + numpy
"""

import re
import struct
from typing import Optional

import numpy as np

# Use the 'regex' library if available (supports \p{L} Unicode properties);
# otherwise fall back to a simplified pattern using the standard 're' module.
try:
    import regex as _re_module   # type: ignore[import]
    _HAS_REGEX = True
except ImportError:
    _re_module = re              # type: ignore[assignment]
    _HAS_REGEX = False

# Pre-compiled regex to split text into initial "words".
if _HAS_REGEX:
    _SPLIT_PATTERN = _re_module.compile(
        r"'s|'t|'re|'ve|'m|'ll|'d"
        r"|[^\r\n\p{L}\p{N}]?\p{L}+"
        r"|\p{N}{1,3}"
        r"| ?[^\s\p{L}\p{N}]+"
        r"|\s+(?!\S)"
        r"|\s+",
        _re_module.UNICODE,
    )
else:
    # Fallback: simplified pattern using standard re (no Unicode property escapes)
    _SPLIT_PATTERN = re.compile(
        r"'s|'t|'re|'ve|'m|'ll|'d"
        r"|[^\r\n\w]?\w+"
        r"|\d{1,3}"
        r"| ?[^\s\w]+"
        r"|\s+(?!\S)"
        r"|\s+",
        re.UNICODE,
    )


# ---------------------------------------------------------------------------
# Token ID constants
# ---------------------------------------------------------------------------

PAD_ID   = 0
UNK_ID   = 1
BOS_ID   = 2
EOS_ID   = 3
FIRST_NORMAL_ID = 4

VOCAB_SIZE_DEFAULT = 16_384
MAX_VOCAB_SIZE     = 65_536   # fits in uint16


# ---------------------------------------------------------------------------
# BPE Tokenizer
# ---------------------------------------------------------------------------

class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer compatible with the TinyFormer-USB device.

    Parameters
    ----------
    vocab  : list[str]   – token strings indexed by token ID
    merges : list[tuple[str, str]]  – BPE merge rules in priority order
    """

    def __init__(
        self,
        vocab: list[str],
        merges: list[tuple[str, str]],
    ) -> None:
        if len(vocab) > MAX_VOCAB_SIZE:
            raise ValueError(f"Vocabulary too large: {len(vocab)} > {MAX_VOCAB_SIZE}")

        self.vocab = vocab
        self.merges = merges

        # Forward map: token string → id
        self._token_to_id: dict[str, int] = {tok: i for i, tok in enumerate(vocab)}
        # Merge priority map: pair → priority index (lower = higher priority)
        self._merge_rank: dict[tuple[str, str], int] = {
            pair: rank for rank, pair in enumerate(merges)
        }

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _tokenize_word(self, word: str) -> list[int]:
        """Apply BPE merges to a single 'word' (character sequence)."""
        # Start with individual UTF-8 bytes rendered as hex-escaped chars
        parts: list[str] = list(word.encode("utf-8").decode("latin-1"))

        while len(parts) > 1:
            # Find the highest-priority (lowest rank) adjacent pair
            best_rank = len(self.merges) + 1
            best_idx = -1
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                rank = self._merge_rank.get(pair, len(self.merges) + 1)
                if rank < best_rank:
                    best_rank = rank
                    best_idx = i

            if best_idx == -1:
                break  # no more merges applicable

            # Apply the merge
            merged = parts[best_idx] + parts[best_idx + 1]
            parts = parts[:best_idx] + [merged] + parts[best_idx + 2:]

        return [self._token_to_id.get(p, UNK_ID) for p in parts]

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        """Encode a string to a list of token IDs."""
        ids: list[int] = []
        if add_bos:
            ids.append(BOS_ID)
        try:
            words = _SPLIT_PATTERN.findall(text)
        except Exception:
            # Fallback: split on whitespace
            words = text.split()
        for word in words:
            ids.extend(self._tokenize_word(word))
        if add_eos:
            ids.append(EOS_ID)
        return ids

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode a list of token IDs back to a string."""
        special = {PAD_ID, UNK_ID, BOS_ID, EOS_ID}
        parts = []
        for tid in ids:
            if skip_special and tid in special:
                continue
            if 0 <= tid < len(self.vocab):
                parts.append(self.vocab[tid])
            else:
                parts.append("<??>")
        raw = "".join(parts)
        # Decode latin-1 back to utf-8
        try:
            return raw.encode("latin-1").decode("utf-8", errors="replace")
        except Exception:
            return raw

    # ------------------------------------------------------------------
    # Serialisation (for on-device Flash storage)
    # ------------------------------------------------------------------

    SERIAL_MAGIC  = b"BPETOK1\x00"
    SERIAL_HEADER = "!8sHI"   # magic(8), vocab_size(2), n_merges(4)

    def to_bytes(self) -> bytes:
        """
        Serialise tokenizer to a compact binary format for on-device storage.

        Format
        ------
        Header:
          [8 B] magic
          [2 B] vocab_size (uint16)
          [4 B] n_merges   (uint32)
        Vocab section:
          For each token: [2 B] token_len (uint16) + [token_len B] token UTF-8
        Merge section:
          For each merge pair: [2B] id_left + [2B] id_right
        """
        buf = bytearray()
        header_size = struct.calcsize(self.SERIAL_HEADER)
        buf.extend(
            struct.pack(self.SERIAL_HEADER, self.SERIAL_MAGIC,
                        len(self.vocab), len(self.merges))
        )
        # Vocab
        for tok in self.vocab:
            tb = tok.encode("utf-8")
            buf.extend(struct.pack("!H", len(tb)))
            buf.extend(tb)
        # Merges stored as (left_id, right_id) pairs
        for left, right in self.merges:
            lid = self._token_to_id.get(left, UNK_ID)
            rid = self._token_to_id.get(right, UNK_ID)
            buf.extend(struct.pack("!HH", lid, rid))
        return bytes(buf)

    @classmethod
    def from_bytes(cls, data: bytes) -> "BPETokenizer":
        """Deserialise a tokenizer previously serialised with ``to_bytes``."""
        header_size = struct.calcsize(cls.SERIAL_HEADER)
        magic, vocab_size, n_merges = struct.unpack_from(cls.SERIAL_HEADER, data, 0)
        if magic != cls.SERIAL_MAGIC:
            raise ValueError("Invalid tokenizer binary: bad magic number")

        offset = header_size
        vocab = []
        for _ in range(vocab_size):
            (tok_len,) = struct.unpack_from("!H", data, offset)
            offset += 2
            tok = data[offset: offset + tok_len].decode("utf-8")
            offset += tok_len
            vocab.append(tok)

        token_to_id = {tok: i for i, tok in enumerate(vocab)}
        merges = []
        for _ in range(n_merges):
            lid, rid = struct.unpack_from("!HH", data, offset)
            offset += 4
            if lid < len(vocab) and rid < len(vocab):
                merges.append((vocab[lid], vocab[rid]))

        return cls(vocab, merges)


# ---------------------------------------------------------------------------
# Minimal vocabulary builder (for demonstration / testing)
# ---------------------------------------------------------------------------

def build_byte_level_vocab() -> tuple[list[str], list[tuple[str, str]]]:
    """
    Return a minimal byte-level vocabulary with no BPE merges.
    All 256 byte values as single tokens + special tokens.
    Useful for unit tests and bootstrap.
    """
    special = ["<pad>", "<unk>", "<bos>", "<eos>"]
    byte_tokens = [chr(i) for i in range(256)]
    vocab = special + byte_tokens
    merges: list[tuple[str, str]] = []
    return vocab, merges


def build_demo_tokenizer() -> "BPETokenizer":
    """Return a tiny demo tokenizer (byte-level, no merges) for testing."""
    vocab, merges = build_byte_level_vocab()
    return BPETokenizer(vocab=vocab, merges=merges)
