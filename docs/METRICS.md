# PagedServe Metrics and Telemetry

PagedServe tracks internal runtime metrics using monotonic clocks and standard telemetry structures.

## Tracked Metrics

### Request Counters
- `requests_total`: Cumulative count of submitted requests.
- `requests_completed`: Successfully finished requests.
- `requests_cancelled`: Requests cancelled by user or client disconnect.
- `requests_failed`: Requests failed due to error or OOM.
- `requests_waiting`: Current number of enqueued waiting requests.
- `requests_running`: Current number of actively scheduled requests.

### Token Counters
- `prompt_tokens_total`: Total prompt tokens processed.
- `generated_tokens_total`: Total output tokens generated.
- `throughput_per_sec`: Generated tokens per second since engine initialization.

### Memory & Cache
- `kv_blocks_total`: Total physical blocks pre-allocated in pool.
- `kv_blocks_used`: Number of physical blocks currently allocated.
- `kv_blocks_free`: Available unallocated physical blocks.
- `utilization`: Fraction of total blocks in use (`used / total`).
- `prefix_cache_hits`: Count of prefix block cache reuses.
- `prefix_cache_misses`: Count of un-cached prefix blocks.
- `prefix_cache_hit_rate`: Ratio of hits to total prefix lookups.

### Latency Distributions
- `ttft_seconds`: Time to First Token statistics (mean, p50, p95, p99).
- `tpot_seconds`: Time Per Output Token statistics during decode iterations.
- `e2e_seconds`: End-to-end request latency statistics.

## Querying Metrics
Metrics are available via the HTTP server at `GET /metrics` or directly via `engine.metrics.to_dict()`.
