#!/usr/bin/env python3
"""Streaming generation example using PagedServe Engine."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pagedserve.config import EngineConfig
from pagedserve.engine.engine import PagedServeEngine
from pagedserve.engine.request import SamplingParams


def main():
    print("Initializing PagedServe Engine...")
    config = EngineConfig(
        model_name_or_path="sshleifer/tiny-gpt2",
        num_blocks=128,
    )
    engine = PagedServeEngine(config=config)

    prompt = "System: You are a helpful assistant.\nUser: What is machine learning?\nAssistant:"
    print(f"Submitting prompt:\n{prompt}\n")
    print("Streaming output tokens: ", end="", flush=True)

    sampling = SamplingParams(max_new_tokens=25, temperature=0.7)
    req_id = engine.submit(prompt=prompt, sampling_params=sampling)

    for event in engine.run_and_stream(req_id):
        print(event.text_delta, end="", flush=True)

    print("\n\n[Stream finished]")


if __name__ == "__main__":
    main()
