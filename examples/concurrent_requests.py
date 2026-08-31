#!/usr/bin/env python3
"""Multi-request continuous batching example using PagedServe Engine."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pagedserve.config import EngineConfig
from pagedserve.engine.engine import PagedServeEngine
from pagedserve.engine.request import SamplingParams


def main():
    print("Initializing PagedServe Engine for concurrent batching...")
    config = EngineConfig(
        model_name_or_path="sshleifer/tiny-gpt2",
        max_num_sequences=4,
        num_blocks=256,
    )
    engine = PagedServeEngine(config=config)

    prompts = [
        "Once upon a time",
        "The future of artificial intelligence is",
        "Deep learning models require",
        "In a galaxy far far away",
    ]

    req_ids = []
    for i, p in enumerate(prompts):
        sampling = SamplingParams(max_new_tokens=15, temperature=0.8)
        rid = engine.submit(prompt=p, sampling_params=sampling, request_id=f"req-{i}")
        req_ids.append(rid)
        print(f"Submitted '{p}' -> {rid}")

    print("\nStepping engine continuously...")
    steps = 0
    while engine.has_active_work:
        engine.step()
        steps += 1

    print(f"\nCompleted all {len(prompts)} requests in {steps} engine steps.")
    for p, rid in zip(prompts, req_ids):
        out = engine.get_output_text(rid)
        print(f"\n[{rid}] Prompt: '{p}'")
        print(f"[{rid}] Output: '{out}'")


if __name__ == "__main__":
    main()
