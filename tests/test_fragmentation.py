from pagedserve.experiments.fragmentation import (
    BlockAllocator,
    ContiguousAllocator,
    run_experiment,
)


def test_fragmentation_allocator_semantics():
    # Contiguous allocator
    contig = ContiguousAllocator(total_size=100)
    assert contig.allocate("r1", 40) is True
    assert contig.allocate("r2", 40) is True
    assert contig.allocate("r3", 30) is False  # Only 20 left
    contig.free("r1")
    # Now holes are [0..40] and [80..100]. Total free = 60.
    assert contig.free_total == 60
    # Request for 50 fails despite 60 free total because largest contiguous hole is 40!
    assert contig.allocate("r4", 50) is False

    # Block allocator
    block = BlockAllocator(total_size=100, block_size=16)
    assert block.allocate("r1", 40) is True  # Needs 3 blocks (48 slots)
    assert block.internal_fragmentation("r1", 40) == (48 - 40) / 48


def test_fragmentation_experiment_run():
    res = run_experiment(
        workload="bimodal",
        total_size=256,
        block_size=16,
        num_requests=20,
        seed=42,
        verbose=False,
    )
    assert "contiguous" in res
    assert "block" in res
    assert res["contiguous"]["success_rate"] <= 1.0
    assert res["block"]["success_rate"] <= 1.0
