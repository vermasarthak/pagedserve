# Contributing to PagedServe

PagedServe is an experimental LLM-serving systems reference with paged KV allocation, continuous-batching simulation and attention-kernel experiments.

## Prerequisites
- Python 3.11+
- PyTorch 2.0+
- Apple Silicon Mac with Metal support (optional, for Metal GPU kernel tests)

## Local Development Workflow

1. **Environment Setup**:
   ```bash
   git clone https://github.com/vermasarthak/pagedserve.git
   cd pagedserve
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .
   pip install pytest ruff transformers
   ```

2. **Linting & Code Style**:
   ```bash
   ruff check .
   ```

3. **Running Test Suite**:
   ```bash
   pytest -v
   ```

## Contribution Invariants
- **Zero KV Leaks**: Every block allocation must have a corresponding deallocation/reclaim on request completion or cancellation.
- **Reference Equivalence**: Any attention kernel optimization must be numerically verified against standard PyTorch attention references.
- **Truthful Scope**: Distinguish simulation components (e.g. single-device tensor partitioning simulation) from distributed multi-node implementations.
