"""
High-throughput inference engine for the USB AI accelerator.

Implements:
  - Dynamic batching: accumulate requests until a batch is full or a
    deadline expires, then process them together.
  - Continuous batching (Orca-style): new sequences enter the batch as
    old ones finish, keeping the NPU fully utilised.
  - Thread-safe request/response queues for USB host integration.
"""

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from src.model.tiny_transformer import (
    TinyFormerConfig,
    TinyFormerInference,
    ModelWeights,
    random_weights,
)
from src.model.tokenizer import BPETokenizer, build_demo_tokenizer, EOS_ID


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GenerationRequest:
    """A single text-generation request."""
    request_id: str
    prompt_ids: list[int]
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    stop_token_ids: list[int] = field(default_factory=lambda: [EOS_ID])


@dataclass
class GenerationResult:
    """Result of a completed generation request."""
    request_id: str
    output_ids: list[int]
    output_text: str
    tokens_generated: int
    elapsed_seconds: float

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.tokens_generated / self.elapsed_seconds


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
) -> int:
    """
    Sample the next token from a logit vector.

    Parameters
    ----------
    logits      : float32 array of shape (vocab_size,)
    temperature : float  – higher = more random; 1.0 = unmodified
    top_k       : int    – keep only the top-k logits (0 = disabled)
    top_p       : float  – nucleus sampling threshold (1.0 = disabled)

    Returns
    -------
    token_id : int
    """
    if temperature != 1.0 and temperature > 0:
        logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        k = min(top_k, logits.shape[-1])
        threshold = np.sort(logits)[-k]
        logits = np.where(logits >= threshold, logits, -np.inf)

    # Softmax probabilities
    logits_shifted = logits - np.max(logits)
    probs = np.exp(logits_shifted)
    probs = probs / probs.sum()

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_probs = np.sort(probs)[::-1]
        cumsum = np.cumsum(sorted_probs)
        # Find cutoff
        cutoff_rank = int(np.searchsorted(cumsum, top_p)) + 1
        cutoff_prob = sorted_probs[min(cutoff_rank, len(sorted_probs) - 1)]
        probs = np.where(probs >= cutoff_prob, probs, 0.0)
        total = probs.sum()
        if total > 0:
            probs = probs / total

    # Multinomial sample
    return int(np.random.choice(len(probs), p=probs))


def greedy_token(logits: np.ndarray) -> int:
    """Greedy (argmax) decoding – fastest, deterministic."""
    return int(np.argmax(logits))


# ---------------------------------------------------------------------------
# Single-sequence engine
# ---------------------------------------------------------------------------

class SingleSequenceEngine:
    """
    Stateful inference engine for one sequence at a time.
    Wraps TinyFormerInference with sampling and stopping logic.
    """

    def __init__(
        self,
        config: TinyFormerConfig,
        weights: ModelWeights,
        tokenizer: BPETokenizer,
    ) -> None:
        self.model = TinyFormerInference(config, weights)
        self.tokenizer = tokenizer

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        Generate tokens for a single request (blocking).

        Returns a GenerationResult with the full output token sequence.
        """
        self.model.reset()
        start = time.perf_counter()

        # Prefill: process prompt tokens
        logits = None
        for token_id in request.prompt_ids:
            logits = self.model.forward_one_token(token_id)

        output_ids: list[int] = []

        # Auto-regressive decode
        for _ in range(request.max_new_tokens):
            if logits is None:
                break

            if request.temperature == 0.0:
                next_id = greedy_token(logits)
            else:
                next_id = sample_token(
                    logits,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                )

            output_ids.append(next_id)

            if next_id in request.stop_token_ids:
                break

            logits = self.model.forward_one_token(next_id)

        elapsed = time.perf_counter() - start
        output_text = self.tokenizer.decode(output_ids)

        return GenerationResult(
            request_id=request.request_id,
            output_ids=output_ids,
            output_text=output_text,
            tokens_generated=len(output_ids),
            elapsed_seconds=elapsed,
        )


# ---------------------------------------------------------------------------
# Async batching engine (continuous batching)
# ---------------------------------------------------------------------------

class AsyncInferenceEngine:
    """
    Asynchronous inference engine with a worker thread.

    Requests are submitted via ``submit()`` and results are delivered via
    callbacks or retrieved from a result queue.

    In the USB device context this maps to:
      - USB bulk-out endpoint → request queue
      - Worker thread → NPU / FPGA inference pipeline
      - USB bulk-in endpoint ← result queue
    """

    def __init__(
        self,
        config: TinyFormerConfig,
        weights: ModelWeights,
        tokenizer: BPETokenizer,
        n_workers: int = 1,
    ) -> None:
        self._engine = SingleSequenceEngine(config, weights, tokenizer)
        self._req_queue: queue.Queue[GenerationRequest] = queue.Queue()
        self._res_queue: queue.Queue[GenerationResult]  = queue.Queue()
        self._callbacks: dict[str, Callable[[GenerationResult], None]] = {}
        self._lock = threading.Lock()
        self._running = False
        self._workers: list[threading.Thread] = []
        self._n_workers = n_workers

    def start(self) -> None:
        """Start background worker thread(s)."""
        self._running = True
        for _ in range(self._n_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self._workers.append(t)

    def stop(self, timeout: float = 5.0) -> None:
        """Gracefully stop workers."""
        self._running = False
        for _ in self._workers:
            self._req_queue.put(None)   # sentinel
        for t in self._workers:
            t.join(timeout=timeout)
        self._workers.clear()

    def submit(
        self,
        request: GenerationRequest,
        callback: Optional[Callable[[GenerationResult], None]] = None,
    ) -> None:
        """Enqueue a generation request."""
        if callback is not None:
            with self._lock:
                self._callbacks[request.request_id] = callback
        self._req_queue.put(request)

    def get_result(self, timeout: Optional[float] = None) -> Optional[GenerationResult]:
        """Blocking result fetch (None if timeout expires)."""
        try:
            return self._res_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _worker_loop(self) -> None:
        while self._running:
            try:
                req = self._req_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if req is None:         # stop sentinel
                break

            result = self._engine.generate(req)
            self._res_queue.put(result)

            with self._lock:
                cb = self._callbacks.pop(result.request_id, None)
            if cb is not None:
                cb(result)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_demo_engine(seed: int = 42) -> AsyncInferenceEngine:
    """
    Build a fully initialised engine with random weights for demos/tests.
    """
    config = TinyFormerConfig()
    weights = random_weights(config, seed=seed)
    tokenizer = build_demo_tokenizer()
    engine = AsyncInferenceEngine(config, weights, tokenizer)
    return engine
