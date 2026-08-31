# PagedServe HTTP API Documentation

PagedServe exposes an OpenAI-compatible subset HTTP API built with FastAPI.

## Launching the Server

```bash
# Using default configuration (sshleifer/tiny-gpt2, CPU/auto)
uvicorn pagedserve.server.api:app --host 127.0.0.1 --port 8000

# Or via custom environment variables:
PAGEDSERVE_MODEL="sshleifer/tiny-gpt2" PAGEDSERVE_NUM_BLOCKS=512 uvicorn pagedserve.server.api:app --host 127.0.0.1 --port 8000
```

## Endpoints

### 1. Health Endpoint
- **URL**: `GET /health`
- **Response**:
```json
{
  "status": "ok",
  "model": "sshleifer/tiny-gpt2",
  "device": "cpu",
  "dtype": "torch.float32",
  "active_requests": 0,
  "waiting_requests": 0
}
```

### 2. Metrics Endpoint
- **URL**: `GET /metrics`
- **Response**: Returns engine telemetry including total requests, token counts, throughput, KV utilization, prefix cache hit rate, and latency percentiles (TTFT, TPOT, E2E).

### 3. Text Completions
- **URL**: `POST /v1/completions`
- **Body**:
```json
{
  "model": "pagedserve",
  "prompt": "Once upon a time",
  "max_tokens": 32,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": false
}
```

#### Streaming Example (`stream=true`):
Emits Server-Sent Events (SSE) data lines terminating with `data: [DONE]`.

```bash
curl -N -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 20, "stream": true}'
```

### 4. Chat Completions
- **URL**: `POST /v1/chat/completions`
- **Body**:
```json
{
  "model": "pagedserve",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
  ],
  "max_tokens": 16,
  "stream": false
}
```
*Note: For models without native chat templates (such as `tiny-gpt2`), PagedServe uses a deterministic fallback formatter (`Role: Content`).*
