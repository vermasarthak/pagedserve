#!/usr/bin/env python3
"""PagedServe throughput benchmark: sequential baseline vs continuous batching engine.

Includes:
- Repeated trials (--trials, --warmup-runs)
- Device synchronization before/after timed blocks
- Exact token prompt generation (validated via tokenizer)
- Statistical summaries (mean, median, stddev, min, max, percentiles)
- TinyLlama-1.1B support via --model or --run-tinyllama

Usage examples
--------------
# Default (sshleifer/tiny-gpt2, fast, ~1 MB model):
python benchmarks/benchmark_throughput.py

# TinyLlama-1.1B (~2.2 GB, requires ~6 GB RAM):
python benchmarks/benchmark_throughput.py --run-tinyllama

# Explicit model:
python benchmarks/benchmark_throughput.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.benchmark_utils import (
    MultiTrialBenchmarkResult,
    TrialStats,
    generate_exact_token_prompts,
    get_hardware_info,
    save_multi_trial_result,
    synchronize_device,
)
from pagedserve.baseline.sequential import SequentialBaseline
from pagedserve.config import EngineConfig
from pagedserve.engine.engine import PagedServeEngine
from pagedserve.engine.request import SamplingParams
from pagedserve.model.loader import ModelLoader

# ---------------------------------------------------------------------------
# Model presets — override defaults for large models that need more blocks
# ---------------------------------------------------------------------------
MODEL_PRESETS = {
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
        "num_blocks": 512,
        "concurrency": 4,
        "prompt_tokens": 64,
        "output_tokens": 32,
        "num_requests": 8,
        "warmup_runs": 1,
        "trials": 3,
        "dtype": "float32",
    },
}


def run_sequential_baseline_trial(loaded_model, prompts, output_tokens, seed=42):
    baseline = SequentialBaseline(loaded_model)
    sampling = SamplingParams(max_new_tokens=output_tokens, temperature=0.0)
    eos_id = loaded_model.tokenizer.eos_token_id
    if eos_id:
        sampling.stop_token_ids.add(eos_id)

    tokenizer = loaded_model.tokenizer
    latencies = []

    synchronize_device(loaded_model.device)
    t0 = time.monotonic()

    for prompt in prompts:
        toks = tokenizer.encode(prompt, add_special_tokens=False) or [eos_id or 0]
        t_req = time.monotonic()
        baseline.generate(toks, sampling, seed=seed)
        synchronize_device(loaded_model.device)
        latencies.append(time.monotonic() - t_req)

    synchronize_device(loaded_model.device)
    total_time = time.monotonic() - t0
    total_out_tokens = sum(output_tokens for _ in prompts)
    return latencies, total_time, total_out_tokens


def run_pagedserve_trial(loaded_model, prompts, output_tokens, config):
    eos_id = loaded_model.tokenizer.eos_token_id
    engine = PagedServeEngine(config=config, loaded_model=loaded_model)

    sampling = SamplingParams(max_new_tokens=output_tokens, temperature=0.0)
    if eos_id:
        sampling.stop_token_ids.add(eos_id)

    req_ids = []
    synchronize_device(loaded_model.device)
    t0 = time.monotonic()

    for i, prompt in enumerate(prompts):
        rid = engine.submit(prompt=prompt, sampling_params=sampling, request_id=f"r{i}")
        req_ids.append(rid)

    while engine.has_active_work:
        engine.step()

    synchronize_device(loaded_model.device)
    total_time = time.monotonic() - t0

    latencies = []
    ttft_list = []
    total_out_tokens = 0
    for rid in req_ids:
        req = engine._requests[rid]
        if req.total_latency:
            latencies.append(req.total_latency)
        if req.ttft:
            ttft_list.append(req.ttft)
        total_out_tokens += req.num_generated_tokens

    return latencies, ttft_list, total_time, total_out_tokens


def main():
    parser = argparse.ArgumentParser(description="PagedServe Throughput Benchmark")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2",
                        help="HuggingFace model name or path (default: sshleifer/tiny-gpt2)")
    parser.add_argument("--run-tinyllama", action="store_true",
                        help="Shortcut: benchmark TinyLlama/TinyLlama-1.1B-Chat-v1.0 "
                             "with tuned defaults (~2.2 GB download, ~6 GB RAM required)")
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--output-dir", default="benchmarks/results")
    args = parser.parse_args()

    # Apply --run-tinyllama shortcut
    if args.run_tinyllama:
        args.model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Apply model-specific preset overrides (only for defaults not explicitly set)
    preset = MODEL_PRESETS.get(args.model, {})
    if preset:
        print(f"[benchmark_throughput] Applying preset overrides for {args.model}")
        # Only override if the user did not set the flag explicitly (check against defaults)
        if args.num_blocks == 256 and "num_blocks" in preset:
            args.num_blocks = preset["num_blocks"]
        if args.concurrency == 8 and "concurrency" in preset:
            args.concurrency = preset["concurrency"]
        if args.prompt_tokens == 32 and "prompt_tokens" in preset:
            args.prompt_tokens = preset["prompt_tokens"]
        if args.output_tokens == 16 and "output_tokens" in preset:
            args.output_tokens = preset["output_tokens"]
        if args.num_requests == 16 and "num_requests" in preset:
            args.num_requests = preset["num_requests"]
        if args.warmup_runs == 2 and "warmup_runs" in preset:
            args.warmup_runs = preset["warmup_runs"]
        if args.trials == 5 and "trials" in preset:
            args.trials = preset["trials"]

    hardware = get_hardware_info()
    print(f"[benchmark_throughput] Loading model: {args.model}")
    dtype = preset.get("dtype", "float32") if preset else "float32"
    loaded = ModelLoader.load(args.model, device=args.device, dtype=dtype)
    print(f"[benchmark_throughput] Device: {loaded.device}, Dtype: {loaded.dtype}")

    prompts = generate_exact_token_prompts(
        loaded.tokenizer, args.num_requests, args.prompt_tokens, seed=args.seed
    )

    config = EngineConfig(
        model_name_or_path=args.model,
        num_blocks=args.num_blocks,
        max_num_sequences=args.concurrency,
        max_batch_tokens=1024,
        max_prefill_tokens_per_step=512,
        enable_prefix_caching=False,
    )

    workload_name = f"p{args.prompt_tokens}_o{args.output_tokens}"

    # ---- PagedServe ----
    print(f"\n[pagedserve] Warming up ({args.warmup_runs} runs)...")
    for _ in range(args.warmup_runs):
        run_pagedserve_trial(loaded, prompts[:2], args.output_tokens, config)

    print(f"[pagedserve] Running {args.trials} measured trials ({args.num_requests} requests, c={args.concurrency})...")
    ps_total_times, ps_req_rates, ps_tok_rates, ps_latencies, ps_ttfts = [], [], [], [], []
    ps_raw = []

    for t in range(args.trials):
        lats, ttfts, total_t, out_toks = run_pagedserve_trial(loaded, prompts, args.output_tokens, config)
        req_rate = args.num_requests / total_t
        tok_rate = out_toks / total_t
        ps_total_times.append(total_t)
        ps_req_rates.append(req_rate)
        ps_tok_rates.append(tok_rate)
        ps_latencies.extend(lats)
        ps_ttfts.extend(ttfts)
        ps_raw.append({
            "trial": t + 1,
            "total_time_s": total_t,
            "requests_per_sec": req_rate,
            "output_tokens_per_sec": tok_rate,
        })
        print(f"  Trial {t+1}: total_time={total_t:.3f}s  req/s={req_rate:.2f}  tok/s={tok_rate:.2f}")

    ps_result = MultiTrialBenchmarkResult(
        model=args.model,
        device=str(loaded.device),
        dtype=str(loaded.dtype),
        mode="pagedserve",
        workload=workload_name,
        concurrency=args.concurrency,
        prompt_tokens=args.prompt_tokens,
        output_tokens=args.output_tokens,
        num_requests=args.num_requests,
        warmup_runs=args.warmup_runs,
        measured_trials=args.trials,
        total_time_stats=TrialStats(ps_total_times),
        requests_per_sec_stats=TrialStats(ps_req_rates),
        tokens_per_sec_stats=TrialStats(ps_tok_rates),
        latency_stats=TrialStats(ps_latencies),
        ttft_stats=TrialStats(ps_ttfts),
        hardware=hardware,
        raw_trials=ps_raw,
    )
    save_multi_trial_result(ps_result, args.output_dir)

    # ---- Sequential Baseline ----
    if not args.skip_baseline:
        print(f"\n[sequential] Warming up ({args.warmup_runs} runs)...")
        for _ in range(args.warmup_runs):
            run_sequential_baseline_trial(loaded, prompts[:2], args.output_tokens, seed=args.seed)

        print(f"[sequential] Running {args.trials} measured trials ({args.num_requests} requests)...")
        seq_total_times, seq_req_rates, seq_tok_rates, seq_latencies = [], [], [], []
        seq_raw = []

        for t in range(args.trials):
            lats, total_t, out_toks = run_sequential_baseline_trial(loaded, prompts, args.output_tokens, seed=args.seed)
            req_rate = args.num_requests / total_t
            tok_rate = out_toks / total_t
            seq_total_times.append(total_t)
            seq_req_rates.append(req_rate)
            seq_tok_rates.append(tok_rate)
            seq_latencies.extend(lats)
            seq_raw.append({
                "trial": t + 1,
                "total_time_s": total_t,
                "requests_per_sec": req_rate,
                "output_tokens_per_sec": tok_rate,
            })
            print(f"  Trial {t+1}: total_time={total_t:.3f}s  req/s={req_rate:.2f}  tok/s={tok_rate:.2f}")

        seq_result = MultiTrialBenchmarkResult(
            model=args.model,
            device=str(loaded.device),
            dtype=str(loaded.dtype),
            mode="sequential",
            workload=workload_name,
            concurrency=1,
            prompt_tokens=args.prompt_tokens,
            output_tokens=args.output_tokens,
            num_requests=args.num_requests,
            warmup_runs=args.warmup_runs,
            measured_trials=args.trials,
            total_time_stats=TrialStats(seq_total_times),
            requests_per_sec_stats=TrialStats(seq_req_rates),
            tokens_per_sec_stats=TrialStats(seq_tok_rates),
            latency_stats=TrialStats(seq_latencies),
            hardware=hardware,
            raw_trials=seq_raw,
        )
        save_multi_trial_result(seq_result, args.output_dir)

    # ---- Summary ----
    print("\n" + "=" * 70)
    print(f"BENCHMARK SUMMARY  model={args.model}  device={loaded.device}")
    print("=" * 70)
    ps_mean_req = sum(ps_req_rates) / len(ps_req_rates)
    ps_mean_tok = sum(ps_tok_rates) / len(ps_tok_rates)
    ps_mean_lat = (sum(ps_latencies) / len(ps_latencies)) if ps_latencies else float("nan")
    print(f"PagedServe (continuous batching, c={args.concurrency}):")
    print(f"  requests/sec:       {ps_mean_req:.2f}")
    print(f"  output tokens/sec:  {ps_mean_tok:.2f}")
    print(f"  mean latency (s):   {ps_mean_lat:.3f}")
    if not args.skip_baseline and seq_req_rates:
        seq_mean_req = sum(seq_req_rates) / len(seq_req_rates)
        seq_mean_tok = sum(seq_tok_rates) / len(seq_tok_rates)
        seq_mean_lat = (sum(seq_latencies) / len(seq_latencies)) if seq_latencies else float("nan")
        print("Sequential baseline:")
        print(f"  requests/sec:       {seq_mean_req:.2f}")
        print(f"  output tokens/sec:  {seq_mean_tok:.2f}")
        print(f"  mean latency (s):   {seq_mean_lat:.3f}")
        speedup = ps_mean_tok / seq_mean_tok if seq_mean_tok > 0 else float("nan")
        print(f"Throughput improvement: {speedup:.2f}×")
    print("=" * 70)


if __name__ == "__main__":
    main()
