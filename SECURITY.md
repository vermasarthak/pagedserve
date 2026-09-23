# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in PagedServe, please report it privately:

- **Email**: `sarthakverma0802@gmail.com`
- **Subject**: `[SECURITY] PagedServe Vulnerability Report`

Please include:
1. Description of the issue (e.g. memory leak, out-of-bounds KV buffer access, unauthorized API access, or SSE stream exhaustion).
2. Minimal reproducible script or request payload.
3. Suggested fix or remediation.

## Operational Boundaries

- PagedServe is an experimental research and systems reference runtime.
- The FastAPI server implementation provides token-based authentication and rate limiting suitable for single-node development testing.
- Metal compute shaders execute bounded threadgroup dispatch routines without dynamic device memory allocation.
