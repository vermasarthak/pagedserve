"""System overhead experiment: separate scheduler & orchestration cost from PyTorch model execution.

Measures:
1. Pure Scheduler step time (queue sorting, token budget accounting, block table updates).
2. Engine state tracking & stream event dispatch overhead.
3. Actual PyTorch model forward pass execution time across different model sizes (tiny-gpt2 vs distilgpt2).
"""

import argparse
import time
import sys
import json
import statistics
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch

from pagedserve.config import EngineConfig
from pagedserve.engine.engine import PagedServeEngine
from pagedserve.engine.request import SamplingParams, InferenceRequest
from pagedserve.memory.kv_cache import KVCacheManager
from pagedserve.model.loader import ModelLoader
from pagedserve.scheduler.policy import FCFSPolicy
from pagedserve.scheduler.scheduler import Scheduler
from benchmarks.benchmark_utils import synchronize_device, get_hardware_info


def measure_pure_scheduler_overhead(num_requests: int = 50, num_steps: int = 100) -> Dict[str, float]:
    """Measure raw Scheduler.schedule() iteration cost without PyTorch model execution."""
    config = EngineConfig(num_blocks=1024, max_num_sequences=64)
    kv_cache = KVCacheManager(block_size=16, num_blocks=1024)
    scheduler = Scheduler(kv_cache_mgr=kv_cache, max_num_sequences=64)

    # Submit mock requests
    for i in range(num_requests):
        req = InferenceRequest(
            request_id=f"mock_{i}",
            prompt=f"prompt_{i}",
            prompt_token_ids=list(range(32)),
            sampling_params=SamplingParams(max_new_tokens=32),
        )
        scheduler.add_request(req)

    times = []
    for _ in range(num_steps):
        t0 = time.monotonic()
        batch = scheduler.schedule()
        t1 = time.monotonic()
        times.append(t1 - t0)
        if batch.is_empty:
            break

    mean_step_ms = (statistics.mean(times) * 1000) if times else 0.0
    return {
        "num_requests": num_requests,
        "total_steps_measured": len(times),
        "mean_scheduler_step_ms": round(mean_step_ms, 4),
        "min_scheduler_step_ms": round(min(times) * 1000, 4) if times else 0.0,
        "max_scheduler_step_ms": round(max(times) * 1000, 4) if times else 0.0,
    }


def measure_engine_breakdown(model_name: str = "sshleifer/tiny-gpt2", device_str: str = "cpu", num_requests: int = 10) -> Dict[str, Any]:
    """Break down step latency into model execution vs orchestration overhead."""
    loaded = ModelLoader.load(model_name, device=device_str)
    config = EngineConfig(model_name_or_path=model_name, num_blocks=256)
    engine = PagedServeEngine(config=config, loaded_model=loaded)

    prompts = ["The future of artificial intelligence"] * num_requests
    sampling = SamplingParams(max_new_tokens=16, temperature=0.0)

    for i, p in enumerate(prompts):
        engine.submit(p, sampling_params=sampling, request_id=f"r{i}")

    step_times = []
    model_times = []

    original_prefill = engine.runner.prefill
    original_decode = engine.runner.decode

    def timed_prefill(*args, **kwargs):
        synchronize_device(loaded.device)
        t0 = time.monotonic()
        res = original_prefill(*args, **kwargs)
        synchronize_device(loaded.device)
        model_times.append(time.monotonic() - t0)
        return res

    def timed_decode(*args, **kwargs):
        synchronize_device(loaded.device)
        t0 = time.monotonic()
        res = original_decode(*args, **kwargs)
        synchronize_device(loaded.device)
        model_times.append(time.monotonic() - t0)
        return res

    engine.runner.prefill = timed_prefill
    engine.runner.decode = timed_decode

    while engine.has_active_work:
        synchronize_device(loaded.device)
        t0 = time.monotonic()
        engine.step()
        synchronize_device(loaded.device)
        step_times.append(time.monotonic() - t0)

    total_step_time = sum(step_times)
    total_model_time = sum(model_times)
    total_orchestration_time = max(0.0, total_step_time - total_model_time)

    return {
        "model": model_name,
        "device": str(loaded.device),
        "total_steps": len(step_times),
        "total_step_time_ms": round(total_step_time * 1000, 2),
        "total_model_forward_ms": round(total_model_time * 1000, 2),
        "total_orchestration_ms": round(total_orchestration_time * 1000, 2),
        "orchestration_pct": round((total_orchestration_time / max(total_step_time, 1e-6)) * 100, 2),
        "mean_model_forward_ms": round((statistics.mean(model_times) * 1000), 4) if model_times else 0.0,
        "mean_step_ms": round((statistics.mean(step_times) * 1000), 4) if step_times else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="PagedServe Overhead Breakdown Experiment")
    parser.add_argument("--output-dir", default="benchmarks/results")
    args = parser.parse_args()

    print("=== System Overhead Experiment ===")
    print("\n1. Measuring Pure Scheduler Overhead (No Model PyTorch execution)...")
    sched_info = measure_pure_scheduler_overhead()
    print(f"  Mean Scheduler Step: {sched_info['mean_scheduler_step_ms']} ms")

    print("\n2. Measuring Step Time Breakdown for Micro Model (sshleifer/tiny-gpt2)...")
    tiny_info = measure_engine_breakdown("sshleifer/tiny-gpt2")
    print(f"  Total Step Time: {tiny_info['total_step_time_ms']} ms")
    print(f"  Model Forward Pass Time: {tiny_info['total_model_forward_ms']} ms")
    print(f"  Orchestration Overhead: {tiny_info['total_orchestration_ms']} ms ({tiny_info['orchestration_pct']}%)")

    print("\n3. Measuring Step Time Breakdown for Small Model (distilgpt2)...")
    distil_info = measure_engine_breakdown("distilgpt2")
    print(f"  Total Step Time: {distil_info['total_step_time_ms']} ms")
    print(f"  Model Forward Pass Time: {distil_info['total_model_forward_ms']} ms")
    print(f"  Orchestration Overhead: {distil_info['total_orchestration_ms']} ms ({distil_info['orchestration_pct']}%)")

    res = {
        "pure_scheduler": sched_info,
        "tiny_gpt2_breakdown": tiny_info,
        "distilgpt2_breakdown": distil_info,
        "hardware": get_hardware_info(),
    }

    out_path = Path(args.output_dir) / "system_overhead_experiment.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
