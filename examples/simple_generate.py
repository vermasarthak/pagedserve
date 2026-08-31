#!/usr/bin/env python3
"""Simple single-request generation example using PagedServe Engine."""

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
        block_size=16,
    )
    engine = PagedServeEngine(config=config)

    prompt = "Hello, my name is"
    print(f"Submitting prompt: '{prompt}'")

    sampling_params = SamplingParams(max_new_tokens=20, temperature=0.7)
    request_id = engine.submit(prompt=prompt, sampling_params=sampling_params)

    print("Running engine steps...")
    engine.run_until_complete(request_id)

    output_text = engine.get_output_text(request_id)
    print(f"\n--- Output ---\n{prompt}{output_text}\n--------------")


if __name__ == "__main__":
    main()
