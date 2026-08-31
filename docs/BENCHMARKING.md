# PagedServe Benchmarking Guide

PagedServe includes reproducible benchmark scripts located in `benchmarks/` to evaluate throughput, latency, and scheduler performance.

## Available Benchmark Scripts

### 1. Throughput Benchmark (`benchmarks/benchmark_throughput.py`)
Compares PagedServe continuous batching against the sequential baseline.

```bash
python benchmarks/benchmark_throughput.py \
  --model sshleifer/tiny-gpt2 \
  --num-requests 20 \
  --prompt-tokens 32 \
  --output-tokens 16
```

### 2. Latency Benchmark (`benchmarks/benchmark_latency.py`)
Measures Time to First Token (TTFT) and per-token generation latency for individual requests.

```bash
python benchmarks/benchmark_latency.py \
  --model sshleifer/tiny-gpt2 \
  --num-requests 10 \
  --prompt-tokens 64 \
  --output-tokens 16
```

### 3. Scheduler Comparison Benchmark (`benchmarks/benchmark_scheduler.py`)
Compares FCFS policy against the experimental `MemoryAwarePolicy` under memory-constrained KV conditions.

```bash
python benchmarks/benchmark_scheduler.py \
  --model sshleifer/tiny-gpt2 \
  --num-requests 20 \
  --num-blocks 64
```

## Output Artifacts
All benchmark runs save structured JSON results to `benchmarks/results/` containing hardware metadata, configuration parameters, requests/sec, tokens/sec, and latency percentiles.
