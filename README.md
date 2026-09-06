# PagedServe

PagedServe is an original experimental LLM inference runtime built from scratch to demonstrate modern systems engineering principles for high-throughput LLM serving.

## Core Subsystems & Features

- **KV Memory Subsystem**: Block-based continuous batching and memory management, including `KVBlock`, `BlockPool`, `BlockTable`, and strict reference counting.
- **Prefix Caching**: Content-addressed deduplication of full blocks using chained SHA-256 cryptographic hashing.
- **Continuous Batching Scheduler**: Iteration-level scheduling with FCFS, chunked prefill, token budgets, and memory-aware admission.
- **Model Execution**: Genuine transformer inference engine integrated via PyTorch (`ModelRunner`, `ModelLoader`, `Sampler`).
- **Sequential Baseline**: Conventional single-request baseline (`SequentialBaseline`) for fair comparative evaluation.
- **Streaming Engine**: Token-by-token async event streaming from engine step iterations.
- **FastAPI HTTP Server**: OpenAI-compatible subset API providing `/health`, `/metrics`, `/v1/completions`, and `/v1/chat/completions` (supporting SSE streaming).
- **Telemetry & Metrics**: Comprehensive internal metric collection tracking TTFT, TPOT, E2E latency, throughput, KV block utilization, and prefix cache hit rate.
- **Benchmark Harness**: Reproducible CLI tools to measure throughput, latency, and scheduler performance under various concurrency levels.
- **Experiments**:
  - **Allocator Fragmentation**: Simulation comparing contiguous vs block-based memory allocation under uniform, bimodal, and heavy-tailed workloads.
  - **Memory-Aware Scheduler**: Experimental pressure-adaptive policy (`MemoryAwarePolicy`) modulating admissions and chunk sizes based on KV pressure.

*Note: Genuine model execution is working. PagedServe currently handles memory virtualization at the metadata layer using Hugging Face's native `DynamicCache` for physical tensors (we are NOT yet implementing custom physical PagedAttention CUDA/Triton kernels).*

## Quick Start

### Installation
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Running the HTTP Server
```bash
uvicorn pagedserve.server.api:app --host 127.0.0.1 --port 8000
```

### Running Examples
```bash
# Simple single-request generation
python examples/simple_generate.py

# Concurrent multi-request continuous batching
python examples/concurrent_requests.py

# Streaming token generation
python examples/streaming_chat.py
```

### Running Benchmarks
```bash
# Throughput benchmark (PagedServe vs Sequential Baseline)
python benchmarks/benchmark_throughput.py --num-requests 20

# Latency benchmark (TTFT and per-token decode latency)
python benchmarks/benchmark_latency.py --num-requests 10

# FCFS vs Memory-Aware Scheduler benchmark
python benchmarks/benchmark_scheduler.py --num-blocks 64

# Run Blockwise Attention Microbenchmark
python benchmarks/benchmark_blockwise_attention.py
```

### Running Experiments
```bash
# Allocator fragmentation simulation
python -m pagedserve.experiments.fragmentation --workload bimodal
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md)
- [KV Cache](docs/KV_CACHE.md)
- [Physical KV Cache](docs/PHYSICAL_KV_CACHE.md)
- [Direct Blockwise Paged Attention](docs/BLOCKWISE_ATTENTION.md)
- [Scheduler](docs/SCHEDULER.md)
- [Model Execution](docs/MODEL_EXECUTION.md)
- [HTTP API](docs/API.md)
- [Benchmarking Guide](docs/BENCHMARKING.md)
- [Benchmark Audit & Claims](docs/BENCHMARK_AUDIT.md)
- [Metrics & Telemetry](docs/METRICS.md)
- [Experiments Guide](docs/EXPERIMENTS.md)
