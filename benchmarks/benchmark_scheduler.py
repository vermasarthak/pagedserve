#!/usr/bin/env python3
"""MemoryAware Policy Audit Benchmark: FCFS vs MemoryAware across multiple workload profiles.

Workload Profiles:
- Workload A (Many Small): 16 small prompts (p=16, o=8)
- Workload B (Few Large): 4 large prompts (p=128, o=32)
- Workload C (Mixed Small/Large): 8 small prompts + 2 large prompts
- Workload D (High Pressure): 12 prompts with KV capacity strictly constrained (num_blocks=16)
"""

import argparse
import time
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from pagedserve.model.loader import ModelLoader
from pagedserve.engine.request import SamplingParams
from pagedserve.config import EngineConfig
from pagedserve.memory.kv_cache import KVCacheManager
from pagedserve.scheduler.policy import FCFSPolicy, MemoryAwarePolicy
from pagedserve.scheduler.scheduler import Scheduler
from pagedserve.engine.engine import PagedServeEngine
from benchmarks.benchmark_utils import (
    MultiTrialBenchmarkResult,
    TrialStats,
    get_hardware_info,
    save_multi_trial_result,
    generate_exact_token_prompts,
    synchronize_device,
)


def run_scheduler_trial(loaded_model, prompts, output_tokens, config, policy, policy_name):
    eos_id = loaded_model.tokenizer.eos_token_id
    kv_cache = KVCacheManager(
        block_size=config.block_size,
        num_blocks=config.num_blocks,
        enable_prefix_caching=False,
    )
    scheduler = Scheduler(
        kv_cache_mgr=kv_cache,
        max_num_sequences=config.max_num_sequences,
        max_batch_tokens=config.max_batch_tokens,
        max_prefill_tokens=config.max_prefill_tokens_per_step,
        prefill_chunk_size=config.max_prefill_tokens_per_step,
        policy=policy,
    )
    engine = PagedServeEngine(
        config=config, loaded_model=loaded_model, scheduler=scheduler, kv_cache_mgr=kv_cache
    )

    sampling = SamplingParams(max_new_tokens=output_tokens, temperature=0.0)
    if eos_id:
        sampling.stop_token_ids.add(eos_id)

    req_ids = []
    synchronize_device(loaded_model.device)
    t0 = time.monotonic()

    for i, prompt in enumerate(prompts):
        try:
            rid = engine.submit(
                prompt=prompt, sampling_params=sampling, request_id=f"{policy_name}_{i}"
            )
            req_ids.append(rid)
        except Exception:
            pass

    while engine.has_active_work:
        engine.step()

    synchronize_device(loaded_model.device)
    total_time = time.monotonic() - t0

    latencies, ttft_list = [], []
    out_tokens = 0
    completed = 0
    for rid in req_ids:
        req = engine._requests.get(rid)
        if req and req.state.name == "FINISHED":
            completed += 1
            out_tokens += req.num_generated_tokens
            if req.total_latency:
                latencies.append(req.total_latency)
            if req.ttft:
                ttft_list.append(req.ttft)

    return total_time, completed, out_tokens, latencies, ttft_list


def audit_policy_comparison(
    loaded_model,
    workload_name: str,
    prompts: List[str],
    output_tokens: int,
    config: EngineConfig,
    trials: int = 5,
    warmup_runs: int = 2,
    output_dir: str = "benchmarks/results",
) -> Dict[str, Any]:
    hardware = get_hardware_info()
    print(f"\n--- AUDITING WORKLOAD: {workload_name} ({len(prompts)} prompts, num_blocks={config.num_blocks}) ---")

    results_summary = {}

    for name, pol in [("FCFS", FCFSPolicy()), ("MemoryAware", MemoryAwarePolicy())]:
        for _ in range(warmup_runs):
            run_scheduler_trial(loaded_model, prompts[:2], output_tokens, config, pol, name)

        times, req_rates, tok_rates, lats, ttfts, completed_counts = [], [], [], [], [], []
        raw = []

        for t in range(trials):
            tot_t, comp, out_toks, trial_lats, trial_ttfts = run_scheduler_trial(
                loaded_model, prompts, output_tokens, config, pol, name
            )
            req_rate = comp / max(tot_t, 1e-6)
            tok_rate = out_toks / max(tot_t, 1e-6)

            times.append(tot_t)
            req_rates.append(req_rate)
            tok_rates.append(tok_rate)
            completed_counts.append(comp)
            lats.extend(trial_lats)
            ttfts.extend(trial_ttfts)

            raw.append({
                "trial": t + 1,
                "total_time_s": tot_t,
                "completed": comp,
                "requests_per_sec": req_rate,
                "output_tokens_per_sec": tok_rate,
            })

        res = MultiTrialBenchmarkResult(
            model=config.model_name_or_path,
            device=str(loaded_model.device),
            dtype=str(loaded_model.dtype),
            mode=f"{name.lower()}_{workload_name}",
            workload=workload_name,
            concurrency=len(prompts),
            prompt_tokens=0,
            output_tokens=output_tokens,
            num_requests=len(prompts),
            warmup_runs=warmup_runs,
            measured_trials=trials,
            total_time_stats=TrialStats(times),
            requests_per_sec_stats=TrialStats(req_rates),
            tokens_per_sec_stats=TrialStats(tok_rates),
            latency_stats=TrialStats(lats),
            ttft_stats=TrialStats(ttfts),
            hardware=hardware,
            raw_trials=raw,
        )
        save_multi_trial_result(res, output_dir)
        results_summary[name] = res.summary()

        print(f"  {name:12s} | Mean Time: {res.total_time_stats.mean:.3f}s (±{res.total_time_stats.stddev:.3f}s) | Req/s: {res.requests_per_sec_stats.mean:.2f} | Completed: {int(sum(completed_counts)/len(completed_counts))}/{len(prompts)}")

    return results_summary


def main():
    parser = argparse.ArgumentParser(description="MemoryAware Policy Audit Benchmark")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-blocks", type=int, default=32)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--output-dir", default="benchmarks/results")
    args = parser.parse_args()

    print(f"[policy_audit] Loading model: {args.model}")
    loaded = ModelLoader.load(args.model, device=args.device)
    tok = loaded.tokenizer

    config = EngineConfig(
        model_name_or_path=args.model,
        num_blocks=args.num_blocks,
        max_num_sequences=8,
        max_batch_tokens=512,
        max_prefill_tokens_per_step=256,
        enable_prefix_caching=False,
    )

    audit_records = {}

    # Workload A: Many Small Requests
    prompts_a = generate_exact_token_prompts(tok, 16, 16, seed=42)
    audit_records["Workload_A_ManySmall"] = audit_policy_comparison(
        loaded, "Workload_A_ManySmall", prompts_a, 8, config, args.trials, args.warmup_runs, args.output_dir
    )

    # Workload B: Few Large Requests
    prompts_b = generate_exact_token_prompts(tok, 4, 128, seed=43)
    audit_records["Workload_B_FewLarge"] = audit_policy_comparison(
        loaded, "Workload_B_FewLarge", prompts_b, 32, config, args.trials, args.warmup_runs, args.output_dir
    )

    # Workload C: Mixed Small and Large
    prompts_c = generate_exact_token_prompts(tok, 8, 16, seed=44) + generate_exact_token_prompts(tok, 2, 128, seed=45)
    audit_records["Workload_C_Mixed"] = audit_policy_comparison(
        loaded, "Workload_C_Mixed", prompts_c, 16, config, args.trials, args.warmup_runs, args.output_dir
    )

    # Workload D: Extreme High Pressure (num_blocks=16)
    config_d = EngineConfig(
        model_name_or_path=args.model,
        num_blocks=16,
        max_num_sequences=8,
        max_batch_tokens=256,
        max_prefill_tokens_per_step=128,
        enable_prefix_caching=False,
    )
    prompts_d = generate_exact_token_prompts(tok, 12, 24, seed=46)
    audit_records["Workload_D_HighPressure"] = audit_policy_comparison(
        loaded, "Workload_D_HighPressure", prompts_d, 16, config_d, args.trials, args.warmup_runs, args.output_dir
    )

    out_audit = Path(args.output_dir) / "scheduler_policy_audit_summary.json"
    with open(out_audit, "w") as f:
        json.dump(audit_records, f, indent=2)
    print(f"\n[POLICY AUDIT COMPLETE] Saved summary to {out_audit}")


if __name__ == "__main__":
    main()
