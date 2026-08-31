#!/usr/bin/env python3
"""Comprehensive Benchmark Suite for PagedServe.

Runs concurrency sweeps, multiple workload profiles, repeated trials,
and model comparisons (sshleifer/tiny-gpt2 vs distilgpt2).
"""

import argparse
import time
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pagedserve.model.loader import ModelLoader
from pagedserve.config import EngineConfig
from pagedserve.engine.engine import PagedServeEngine
from pagedserve.engine.request import SamplingParams
from pagedserve.baseline.sequential import SequentialBaseline
from benchmarks.benchmark_utils import (
    MultiTrialBenchmarkResult,
    TrialStats,
    get_hardware_info,
    save_multi_trial_result,
    generate_exact_token_prompts,
    synchronize_device,
)

WORKLOAD_PROFILES = {
    "MICRO": {"prompt_tokens": 16, "output_tokens": 8},
    "SHORT": {"prompt_tokens": 32, "output_tokens": 16},
    "MEDIUM": {"prompt_tokens": 128, "output_tokens": 32},
}


def run_benchmark_matrix(
    models=("sshleifer/tiny-gpt2", "distilgpt2"),
    concurrencies=(1, 2, 4, 8),
    workloads=("MICRO", "SHORT"),
    trials=3,
    warmup_runs=1,
    output_dir="benchmarks/results",
):
    hardware = get_hardware_info()
    print("=== Starting Comprehensive Benchmark Suite ===")
    print(f"Hardware: Python {hardware['python_version']} | Torch {hardware['torch_version']} | MPS: {hardware['mps_available']}")

    summary_records = []

    for model_name in models:
        print(f"\n==========================================")
        print(f"LOADING MODEL: {model_name}")
        print(f"==========================================")
        try:
            loaded = ModelLoader.load(model_name, dtype="float32")
        except Exception as e:
            print(f"[SKIP] Could not load model '{model_name}': {e}")
            continue

        for w_name in workloads:
            profile = WORKLOAD_PROFILES[w_name]
            p_toks = profile["prompt_tokens"]
            o_toks = profile["output_tokens"]
            num_requests = 16

            prompts = generate_exact_token_prompts(
                loaded.tokenizer, num_requests, p_toks, seed=42
            )

            for c in concurrencies:
                print(f"\n--- Model: {model_name} | Workload: {w_name} (p={p_toks}, o={o_toks}) | Concurrency: {c} ---")

                config = EngineConfig(
                    model_name_or_path=model_name,
                    num_blocks=256,
                    max_num_sequences=c,
                    max_batch_tokens=1024,
                    max_prefill_tokens_per_step=512,
                    enable_prefix_caching=False,
                )

                # Warmup
                for _ in range(warmup_runs):
                    engine = PagedServeEngine(config=config, loaded_model=loaded)
                    sampling = SamplingParams(max_new_tokens=o_toks, temperature=0.0)
                    for i, p in enumerate(prompts[:c]):
                        engine.submit(p, sampling_params=sampling, request_id=f"w_{i}")
                    while engine.has_active_work:
                        engine.step()

                # Measured Trials
                ps_times, ps_req_rates, ps_tok_rates, ps_lats, ps_ttfts = [], [], [], [], []
                for t in range(trials):
                    engine = PagedServeEngine(config=config, loaded_model=loaded)
                    sampling = SamplingParams(max_new_tokens=o_toks, temperature=0.0)
                    req_ids = [engine.submit(p, sampling_params=sampling, request_id=f"r_{i}") for i, p in enumerate(prompts)]
                    
                    synchronize_device(loaded.device)
                    t0 = time.monotonic()
                    while engine.has_active_work:
                        engine.step()
                    synchronize_device(loaded.device)
                    total_t = time.monotonic() - t0

                    req_rate = num_requests / total_t
                    out_toks = sum(engine._requests[rid].num_generated_tokens for rid in req_ids)
                    tok_rate = out_toks / total_t

                    ps_times.append(total_t)
                    ps_req_rates.append(req_rate)
                    ps_tok_rates.append(tok_rate)
                    ps_lats.extend([engine._requests[rid].total_latency for rid in req_ids if engine._requests[rid].total_latency])
                    ps_ttfts.extend([engine._requests[rid].ttft for rid in req_ids if engine._requests[rid].ttft])

                ps_res = MultiTrialBenchmarkResult(
                    model=model_name,
                    device=str(loaded.device),
                    dtype=str(loaded.dtype),
                    mode="pagedserve",
                    workload=w_name,
                    concurrency=c,
                    prompt_tokens=p_toks,
                    output_tokens=o_toks,
                    num_requests=num_requests,
                    warmup_runs=warmup_runs,
                    measured_trials=trials,
                    total_time_stats=TrialStats(ps_times),
                    requests_per_sec_stats=TrialStats(ps_req_rates),
                    tokens_per_sec_stats=TrialStats(ps_tok_rates),
                    latency_stats=TrialStats(ps_lats),
                    ttft_stats=TrialStats(ps_ttfts),
                    hardware=hardware,
                )
                save_multi_trial_result(ps_res, output_dir)
                summary_records.append(ps_res.summary())
                print(f"  PagedServe c={c}: req/s={ps_res.requests_per_sec_stats.mean:.2f} | tok/s={ps_res.tokens_per_sec_stats.mean:.2f} | latency_mean={ps_res.latency_stats.mean:.4f}s")

    out_summary = Path(output_dir) / "benchmark_suite_summary.json"
    with open(out_summary, "w") as f:
        json.dump(summary_records, f, indent=2)
    print(f"\n[SUITE COMPLETE] Saved summary matrix to {out_summary}")


def main():
    parser = argparse.ArgumentParser(description="PagedServe Full Benchmark Suite")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--output-dir", default="benchmarks/results")
    args = parser.parse_args()

    run_benchmark_matrix(
        trials=args.trials,
        warmup_runs=args.warmup_runs,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
