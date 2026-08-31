"""Allocator fragmentation simulation: contiguous vs fixed-size block allocator.

This experiment demonstrates allocation/fragmentation tradeoffs under variable
sequence sizes WITHOUT involving real transformer tensors.
"""

import argparse
import datetime
import json
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from pathlib import Path


@dataclass
class Allocation:
    request_id: str
    start: int
    size: int
    alive: bool = True


class ContiguousAllocator:
    """Simulates a fixed-size memory arena with contiguous first-fit allocation."""

    def __init__(self, total_size: int) -> None:
        self.total_size = total_size
        self._allocations: List[Allocation] = []

    def _free_holes(self) -> List[Tuple[int, int]]:
        prev_end = 0
        holes = []
        for a in sorted(self._allocations, key=lambda x: x.start):
            if not a.alive:
                continue
            if a.start > prev_end:
                holes.append((prev_end, a.start - prev_end))
            prev_end = max(prev_end, a.start + a.size)
        if prev_end < self.total_size:
            holes.append((prev_end, self.total_size - prev_end))
        return holes

    def allocate(self, request_id: str, size: int) -> bool:
        for start, hole_size in self._free_holes():
            if hole_size >= size:
                self._allocations.append(Allocation(request_id, start, size))
                return True
        return False

    def free(self, request_id: str) -> None:
        for a in self._allocations:
            if a.request_id == request_id and a.alive:
                a.alive = False
                return

    @property
    def used(self) -> int:
        return sum(a.size for a in self._allocations if a.alive)

    @property
    def free_total(self) -> int:
        return self.total_size - self.used

    @property
    def external_fragmentation(self) -> float:
        holes = self._free_holes()
        if not holes:
            return 0.0
        hole_sizes = [h[1] for h in holes]
        total_free = sum(hole_sizes)
        if total_free == 0:
            return 0.0
        largest = max(hole_sizes)
        unusable = total_free - largest
        return unusable / self.total_size

    @property
    def utilization(self) -> float:
        return self.used / self.total_size


class BlockAllocator:
    """Simulates a block-based allocator with fixed page size."""

    def __init__(self, total_size: int, block_size: int) -> None:
        self.total_size = total_size
        self.block_size = block_size
        self.total_blocks = total_size // block_size
        self._free_blocks: int = self.total_blocks
        self._allocations: Dict[str, int] = {}

    def blocks_needed(self, tokens: int) -> int:
        return (tokens + self.block_size - 1) // self.block_size

    def allocate(self, request_id: str, tokens: int) -> bool:
        needed = self.blocks_needed(tokens)
        if needed > self._free_blocks:
            return False
        self._allocations[request_id] = needed
        self._free_blocks -= needed
        return True

    def free(self, request_id: str) -> None:
        if request_id in self._allocations:
            self._free_blocks += self._allocations.pop(request_id)

    @property
    def used_blocks(self) -> int:
        return self.total_blocks - self._free_blocks

    @property
    def utilization(self) -> float:
        return self.used_blocks / self.total_blocks

    def internal_fragmentation(self, request_id: str, actual_tokens: int) -> float:
        if request_id not in self._allocations:
            return 0.0
        allocated_slots = self._allocations[request_id] * self.block_size
        wasted = allocated_slots - actual_tokens
        return wasted / allocated_slots if allocated_slots > 0 else 0.0


@dataclass
class WorkloadEvent:
    request_id: str
    arrival_tick: int
    lifetime_ticks: int
    tokens: int


def generate_uniform_workload(n: int, total_size: int, seed: int = 42) -> List[WorkloadEvent]:
    rng = random.Random(seed)
    events = []
    for i in range(n):
        arrival = rng.randint(0, n)
        lifetime = rng.randint(1, 10)
        tokens = rng.randint(total_size // 20, total_size // 5)
        events.append(WorkloadEvent(f"req_{i}", arrival, lifetime, tokens))
    return sorted(events, key=lambda e: e.arrival_tick)


def generate_bimodal_workload(n: int, total_size: int, seed: int = 42) -> List[WorkloadEvent]:
    rng = random.Random(seed)
    events = []
    for i in range(n):
        arrival = rng.randint(0, n * 2)
        lifetime = rng.randint(1, 15)
        if rng.random() < 0.6:
            tokens = rng.randint(total_size // 50, total_size // 10)
        else:
            tokens = rng.randint(total_size // 5, total_size // 2)
        events.append(WorkloadEvent(f"req_{i}", arrival, lifetime, tokens))
    return sorted(events, key=lambda e: e.arrival_tick)


def generate_heavy_tail_workload(n: int, total_size: int, seed: int = 42) -> List[WorkloadEvent]:
    rng = random.Random(seed)
    events = []
    for i in range(n):
        arrival = rng.randint(0, n)
        lifetime = rng.randint(1, 20)
        u = rng.random()
        tokens = max(1, int(total_size * (u**3)))
        events.append(WorkloadEvent(f"req_{i}", arrival, lifetime, tokens))
    return sorted(events, key=lambda e: e.arrival_tick)


@dataclass
class SimResult:
    allocator_type: str
    workload: str
    total_size: int
    block_size: Optional[int]
    num_requests: int
    allocation_successes: int = 0
    allocation_failures: int = 0
    failure_despite_free_capacity: int = 0
    peak_utilization: float = 0.0
    utilization_samples: List[float] = field(default_factory=list)
    internal_fragmentation_samples: List[float] = field(default_factory=list)
    external_fragmentation_samples: List[float] = field(default_factory=list)

    def summary(self) -> Dict:
        import statistics

        d = {
            "allocator": self.allocator_type,
            "workload": self.workload,
            "total_size": self.total_size,
            "block_size": self.block_size,
            "num_requests": self.num_requests,
            "success_rate": round(self.allocation_successes / max(1, self.num_requests), 4),
            "allocation_failures": self.allocation_failures,
            "failure_despite_free_capacity": self.failure_despite_free_capacity,
            "peak_utilization": round(self.peak_utilization, 4),
            "mean_utilization": round(
                statistics.mean(self.utilization_samples) if self.utilization_samples else 0.0, 4
            ),
        }
        if self.internal_fragmentation_samples:
            d["mean_internal_fragmentation"] = round(
                statistics.mean(self.internal_fragmentation_samples), 4
            )
        if self.external_fragmentation_samples:
            d["mean_external_fragmentation"] = round(
                statistics.mean(self.external_fragmentation_samples), 4
            )
        return d


def simulate_contiguous(
    events: List[WorkloadEvent], total_size: int, workload_name: str
) -> SimResult:
    alloc = ContiguousAllocator(total_size)
    result = SimResult(
        allocator_type="contiguous",
        workload=workload_name,
        total_size=total_size,
        block_size=None,
        num_requests=len(events),
    )
    active: Dict[str, int] = {}

    max_tick = max((e.arrival_tick + e.lifetime_ticks for e in events), default=1)
    event_map: Dict[int, List[WorkloadEvent]] = {}
    for e in events:
        event_map.setdefault(e.arrival_tick, []).append(e)

    for tick in range(max_tick + 1):
        expired = [rid for rid, deadline in active.items() if deadline <= tick]
        for rid in expired:
            alloc.free(rid)
            del active[rid]

        for ev in event_map.get(tick, []):
            free_before = alloc.free_total
            success = alloc.allocate(ev.request_id, ev.tokens)
            if success:
                result.allocation_successes += 1
                active[ev.request_id] = tick + ev.lifetime_ticks
            else:
                result.allocation_failures += 1
                if free_before >= ev.tokens:
                    result.failure_despite_free_capacity += 1

        util = alloc.utilization
        result.utilization_samples.append(util)
        result.peak_utilization = max(result.peak_utilization, util)
        result.external_fragmentation_samples.append(alloc.external_fragmentation)

    return result


def simulate_block(
    events: List[WorkloadEvent], total_size: int, block_size: int, workload_name: str
) -> SimResult:
    alloc = BlockAllocator(total_size, block_size)
    result = SimResult(
        allocator_type="block",
        workload=workload_name,
        total_size=total_size,
        block_size=block_size,
        num_requests=len(events),
    )
    active: Dict[str, Tuple[int, int]] = {}

    max_tick = max((e.arrival_tick + e.lifetime_ticks for e in events), default=1)
    event_map: Dict[int, List[WorkloadEvent]] = {}
    for e in events:
        event_map.setdefault(e.arrival_tick, []).append(e)

    for tick in range(max_tick + 1):
        expired = [rid for rid, (deadline, _) in active.items() if deadline <= tick]
        for rid in expired:
            alloc.free(rid)
            del active[rid]

        for ev in event_map.get(tick, []):
            success = alloc.allocate(ev.request_id, ev.tokens)
            if success:
                result.allocation_successes += 1
                active[ev.request_id] = (tick + ev.lifetime_ticks, ev.tokens)
                frag = alloc.internal_fragmentation(ev.request_id, ev.tokens)
                result.internal_fragmentation_samples.append(frag)
            else:
                result.allocation_failures += 1

        util = alloc.utilization
        result.utilization_samples.append(util)
        result.peak_utilization = max(result.peak_utilization, util)

    return result


def run_experiment(
    workload: str = "bimodal",
    total_size: int = 1024,
    block_size: int = 16,
    num_requests: int = 100,
    seed: int = 42,
    output_dir: str = "benchmarks/results",
    verbose: bool = True,
) -> Dict:
    generators = {
        "uniform": generate_uniform_workload,
        "bimodal": generate_bimodal_workload,
        "heavy_tail": generate_heavy_tail_workload,
    }
    gen_fn = generators.get(workload, generate_bimodal_workload)
    events = gen_fn(num_requests, total_size, seed=seed)

    contig_result = simulate_contiguous(events, total_size, workload)
    block_result = simulate_block(events, total_size, block_size, workload)

    results = {
        "config": {
            "workload": workload,
            "total_size": total_size,
            "block_size": block_size,
            "num_requests": num_requests,
            "seed": seed,
        },
        "contiguous": contig_result.summary(),
        "block": block_result.summary(),
    }

    if verbose:
        print(f"\n=== Fragmentation Experiment: {workload} workload ===")
        print(f"\nContiguous Allocator:")
        for k, v in contig_result.summary().items():
            print(f"  {k}: {v}")
        print(f"\nBlock Allocator (block_size={block_size}):")
        for k, v in block_result.summary().items():
            print(f"  {k}: {v}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    fpath = out_path / f"fragmentation_{workload}_{ts}.json"
    with open(fpath, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Allocator Fragmentation Experiment")
    parser.add_argument(
        "--workload", choices=["uniform", "bimodal", "heavy_tail"], default="bimodal"
    )
    parser.add_argument("--total-size", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="benchmarks/results")
    args = parser.parse_args()

    run_experiment(
        workload=args.workload,
        total_size=args.total_size,
        block_size=args.block_size,
        num_requests=args.num_requests,
        seed=args.seed,
        output_dir=args.output_dir,
        verbose=True,
    )
