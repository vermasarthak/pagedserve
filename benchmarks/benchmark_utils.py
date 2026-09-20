"""Shared utilities for PagedServe benchmark scripts and device synchronization."""

import datetime
import json
import platform
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


def synchronize_device(device: torch.device | str) -> None:
    """Synchronize accelerator execution queue before/after timing sections.
    
    - CUDA: torch.cuda.synchronize()
    - MPS: torch.mps.synchronize() (if available in PyTorch build)
    - CPU: no-op
    """
    dev_str = str(device).lower()
    if "cuda" in dev_str and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif "mps" in dev_str and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        try:
            torch.mps.synchronize()
        except Exception:
            pass


@dataclass
class TrialStats:
    """Statistical summary across multiple benchmark trials."""

    values: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float | None:
        return statistics.mean(self.values) if self.values else None

    @property
    def median(self) -> float | None:
        return statistics.median(self.values) if self.values else None

    @property
    def stddev(self) -> float | None:
        return statistics.stdev(self.values) if len(self.values) > 1 else 0.0

    @property
    def min_val(self) -> float | None:
        return min(self.values) if self.values else None

    @property
    def max_val(self) -> float | None:
        return max(self.values) if self.values else None

    def percentile(self, p: float) -> float | None:
        if not self.values:
            return None
        sorted_vals = sorted(self.values)
        n = len(sorted_vals)
        idx = max(0, min(n - 1, int(p * n / 100)))
        return sorted_vals[idx]

    def to_dict(self) -> dict[str, Any]:
        if not self.values:
            return {
                "count": 0,
                "mean": None,
                "median": None,
                "stddev": None,
                "min": None,
                "max": None,
            }
        return {
            "count": self.count,
            "mean": round(self.mean, 4) if self.mean is not None else None,
            "median": round(self.median, 4) if self.median is not None else None,
            "stddev": round(self.stddev, 4) if self.stddev is not None else None,
            "min": round(self.min_val, 4) if self.min_val is not None else None,
            "max": round(self.max_val, 4) if self.max_val is not None else None,
            "p50": round(self.percentile(50), 4) if self.percentile(50) is not None else None,
            "p95": round(self.percentile(95), 4) if self.percentile(95) is not None else None,
            "p99": round(self.percentile(99), 4) if self.percentile(99) is not None else None,
        }


@dataclass
class MultiTrialBenchmarkResult:
    """Aggregated results across multiple repeated trials of a benchmark."""

    model: str
    device: str
    dtype: str
    mode: str
    workload: str
    concurrency: int
    prompt_tokens: int
    output_tokens: int
    num_requests: int
    warmup_runs: int
    measured_trials: int
    total_time_stats: TrialStats = field(default_factory=TrialStats)
    requests_per_sec_stats: TrialStats = field(default_factory=TrialStats)
    tokens_per_sec_stats: TrialStats = field(default_factory=TrialStats)
    latency_stats: TrialStats = field(default_factory=TrialStats)
    ttft_stats: TrialStats = field(default_factory=TrialStats)
    hardware: dict[str, Any] = field(default_factory=dict)
    raw_trials: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat()
    )

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "model": self.model,
            "device": self.device,
            "dtype": self.dtype,
            "workload": self.workload,
            "concurrency": self.concurrency,
            "num_requests": self.num_requests,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "warmup_runs": self.warmup_runs,
            "measured_trials": self.measured_trials,
            "total_time_s": self.total_time_stats.to_dict(),
            "requests_per_sec": self.requests_per_sec_stats.to_dict(),
            "output_tokens_per_sec": self.tokens_per_sec_stats.to_dict(),
            "latency_s": self.latency_stats.to_dict(),
            "ttft_s": self.ttft_stats.to_dict(),
            "timestamp": self.timestamp,
        }


def get_hardware_info() -> dict[str, Any]:
    """Collect hardware metadata for reproducible benchmark records."""
    import psutil

    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
        )
    return info


def save_multi_trial_result(
    result: MultiTrialBenchmarkResult, output_dir: str = "benchmarks/results"
) -> str:
    """Save benchmark result as structured JSON."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    fname = f"{result.mode}_{result.model.replace('/', '_')}_{result.workload}_{result.concurrency}c_{ts}.json"
    fpath = out_path / fname

    full_dict = result.summary()
    full_dict["hardware"] = result.hardware
    full_dict["raw_trials"] = result.raw_trials

    with open(fpath, "w") as f:
        json.dump(full_dict, f, indent=2)
    return str(fpath)


def generate_exact_token_prompts(
    tokenizer: Any,
    num_prompts: int,
    target_token_count: int,
    seed: int = 42,
) -> list[str]:
    """Generate prompts that tokenize to EXACTLY target_token_count tokens."""
    import random

    rng = random.Random(seed)
    words = [
        "the", "a", "an", "is", "are", "was", "were", "has", "have", "had",
        "be", "been", "being", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "of", "in", "to", "for",
        "on", "with", "at", "by", "from", "up", "about", "into", "through",
        "system", "model", "data", "result", "output", "input", "value",
        "function", "class", "variable", "method", "object", "return",
    ]

    prompts = []
    for _ in range(num_prompts):
        tokens: list[int] = []
        words_buf: list[str] = []
        while len(tokens) < target_token_count:
            w = rng.choice(words)
            words_buf.append(w)
            text = " ".join(words_buf)
            tokens = tokenizer.encode(text, add_special_tokens=False)

        # Truncate tokens to exact count and decode back
        exact_tokens = tokens[:target_token_count]
        exact_text = tokenizer.decode(exact_tokens, skip_special_tokens=True)
        prompts.append(exact_text)

    return prompts
