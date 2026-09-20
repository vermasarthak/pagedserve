#!/usr/bin/env python3
"""PagedServe latency benchmark: measure TTFT and per-token latency."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.benchmark_utils import (
    BenchmarkResult,
    generate_synthetic_prompts,
    get_hardware_info,
    save_result,
)
from pagedserve.config import EngineConfig
from pagedserve.engine.engine import PagedServeEngine
from pagedserve.engine.request import SamplingParams
from pagedserve.model.loader import ModelLoader


def main():
    parser = argparse.ArgumentParser(description="PagedServe Latency Benchmark")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--num-requests", type=int, default=10)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="benchmarks/results")
    args = parser.parse_args()

    hardware = get_hardware_info()
    print(f"[latency_bench] Loading model: {args.model}")
    loaded = ModelLoader.load(args.model, device=args.device)
    print(f"[latency_bench] Device: {loaded.device}")

    prompts = generate_synthetic_prompts(args.num_requests, args.prompt_tokens, seed=args.seed)

    config = EngineConfig(
        model_name_or_path=args.model,
        num_blocks=args.num_blocks,
        max_num_sequences=4,
        max_batch_tokens=512,
        max_prefill_tokens_per_step=256,
        enable_prefix_caching=False,
    )
    engine = PagedServeEngine(config=config, loaded_model=loaded)
    eos_id = loaded.tokenizer.eos_token_id

    sampling = SamplingParams(max_new_tokens=args.output_tokens, temperature=0.0)
    if eos_id:
        sampling.stop_token_ids.add(eos_id)

    print("[latency_bench] Warming up...")
    warm_id = engine.submit(prompts[0], sampling_params=sampling)
    while engine.has_active_work:
        engine.step()

    print(f"[latency_bench] Measuring {args.num_requests} requests one at a time...")
    latencies = []
    ttft_list = []

    for i, prompt in enumerate(prompts):
        req_id = engine.submit(prompt, sampling_params=sampling, request_id=f"lat_{i}")
        t0 = time.monotonic()
        while engine._requests[req_id].state.name not in ("FINISHED", "CANCELLED", "FAILED"):
            engine.step()
        elapsed = time.monotonic() - t0
        req = engine._requests[req_id]
        latencies.append(elapsed)
        if req.ttft:
            ttft_list.append(req.ttft)

    result = BenchmarkResult(
        model=args.model,
        device=str(loaded.device),
        dtype=str(loaded.dtype),
        mode="pagedserve_latency",
        concurrency=1,
        prompt_tokens=args.prompt_tokens,
        output_tokens=args.output_tokens,
        num_requests=args.num_requests,
        total_time_s=sum(latencies),
        requests_per_sec=args.num_requests / sum(latencies),
        output_tokens_per_sec=(args.num_requests * args.output_tokens) / sum(latencies),
        latencies_s=latencies,
        ttft_s=ttft_list,
        hardware=hardware,
    )
    print(
        f"  mean_latency={result.mean_latency:.3f}s  p50={result.p50_latency:.3f}s  p95={result.p95_latency:.3f}s"
    )
    if result.mean_ttft:
        print(f"  mean_ttft={result.mean_ttft:.4f}s")
    path = save_result(result, args.output_dir)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
