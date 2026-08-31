# PagedServe Experiments

PagedServe includes two major systems experimentation suites:

## 1. Allocator Fragmentation Experiment (`pagedserve/experiments/fragmentation.py`)
Simulates memory management under variable-length workloads without PyTorch tensor overhead.

### Purpose
Compares a contiguous arena allocator against a fixed-size block allocator to demonstrate external vs internal fragmentation tradeoffs.

### Metrics Tracked
- **Allocation Success Rate**: Percentage of requests successfully allocated.
- **Failures Despite Free Capacity**: Contiguous allocator failures caused by external fragmentation when total free memory is sufficient but non-contiguous.
- **External Fragmentation**: Unusable memory locked between allocated contiguous regions.
- **Internal Fragmentation**: Unused tail capacity in the final block of a page-allocated sequence.

### Running the Experiment
```bash
python -m pagedserve.experiments.fragmentation --workload bimodal --seed 42
```

---

## 2. Memory-Aware Scheduling Experiment (`pagedserve/scheduler/policy.py`)
Introduces an experimental `MemoryAwarePolicy` that adapts scheduling decisions based on KV memory pressure.

### Pressure Thresholds
- **LOW (< 60% KV utilization)**: Standard FCFS admission and max prefill chunk size.
- **MEDIUM (60% - 85% KV utilization)**: Modestly restricts new request admissions and halves prefill chunk sizes to avoid memory spikes.
- **HIGH (> 85% KV utilization)**: Strongly restricts admissions to small prompts and prioritizes requests close to completion to free KV memory quickly.

### Starvation Prevention
Waiting requests record iteration counters. Once wait iterations exceed `starvation_penalty_iters`, aging bonuses elevate priority to guarantee eventual execution.

### Benchmark Comparison
```bash
python benchmarks/benchmark_scheduler.py --num-blocks 64
```
