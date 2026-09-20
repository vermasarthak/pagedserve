"""Continuous batching scheduler managing request queues, prefill/decode budgets, and chunking."""


from pagedserve.engine.request import InferenceRequest
from pagedserve.engine.state import RequestState
from pagedserve.errors import InvalidRequestError, InvalidRequestStateError
from pagedserve.memory.kv_cache import KVCacheManager
from pagedserve.scheduler.batch import ScheduledItem, SchedulerBatch, WorkType
from pagedserve.scheduler.policy import FCFSPolicy, SchedulingPolicy


class Scheduler:
    """Iteration-level Continuous Batching Scheduler.
    
    Key Responsibilities:
    1. Continuous Batching: Retires finished sequences and admits new waiting sequences
       at every single iteration step without waiting for an entire batch to complete.
    2. Prefill vs. Decode Separation: Explicitly partitions batch items into compute-bound
       prompt prefills and memory-bound auto-regressive decodes.
    3. Chunked Prefill: Bounds prompt chunk sizes (e.g. 512 tokens) so long prompts do not
       starve active decode requests, maintaining steady Time-Per-Output-Token (TPOT).
    4. Memory-Aware Admission: Consults KVCacheManager before admitting requests; defers
       admission gracefully when KV blocks are temporarily exhausted.
    5. Clean Resource Deallocation: Automatically frees all physical KV blocks upon request
       completion, cancellation, or failure.
    """

    def __init__(
        self,
        kv_cache_mgr: KVCacheManager,
        max_num_sequences: int = 32,
        max_batch_tokens: int = 2048,
        max_prefill_tokens: int = 512,
        prefill_chunk_size: int = 512,
        policy: SchedulingPolicy | None = None,
    ):
        if max_num_sequences <= 0:
            raise ValueError(f"max_num_sequences must be positive, got {max_num_sequences}")
        if max_batch_tokens <= 0:
            raise ValueError(f"max_batch_tokens must be positive, got {max_batch_tokens}")
        if max_prefill_tokens <= 0:
            raise ValueError(f"max_prefill_tokens must be positive, got {max_prefill_tokens}")
        if prefill_chunk_size <= 0:
            raise ValueError(f"prefill_chunk_size must be positive, got {prefill_chunk_size}")

        self.kv_cache_mgr: KVCacheManager = kv_cache_mgr
        self.max_num_sequences: int = max_num_sequences
        self.max_batch_tokens: int = max_batch_tokens
        self.max_prefill_tokens: int = max_prefill_tokens
        self.prefill_chunk_size: int = prefill_chunk_size
        self.policy: SchedulingPolicy = policy if policy is not None else FCFSPolicy()

        # Request state queues
        self._waiting: list[InferenceRequest] = []
        self._running: dict[str, InferenceRequest] = {}
        self._completed: list[InferenceRequest] = []
        self._cancelled: list[InferenceRequest] = []
        self._failed: list[InferenceRequest] = []

    # --- Queue Telemetry ---

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        return len(self._running)

    @property
    def num_completed(self) -> int:
        return len(self._completed)

    @property
    def num_cancelled(self) -> int:
        return len(self._cancelled)

    @property
    def num_failed(self) -> int:
        return len(self._failed)

    @property
    def waiting_requests(self) -> list[InferenceRequest]:
        return list(self._waiting)

    @property
    def running_requests(self) -> dict[str, InferenceRequest]:
        return dict(self._running)

    @property
    def completed_requests(self) -> list[InferenceRequest]:
        return list(self._completed)

    # --- Request Enqueue & Cancellation ---

    def add_request(self, request: InferenceRequest) -> None:
        """Enqueue a new inference request into the waiting queue.
        
        Raises:
            InvalidRequestStateError: if request is not in WAITING state.
            InvalidRequestError: if request ID already exists in scheduler.
        """
        if request.state != RequestState.WAITING:
            raise InvalidRequestStateError(
                f"Cannot enqueue request '{request.request_id}' with state '{request.state}'"
            )
        if request.request_id in self._running or any(r.request_id == request.request_id for r in self._waiting):
            raise InvalidRequestError(f"Duplicate request ID '{request.request_id}'")

        self._waiting.append(request)

    def cancel_request(self, request_id: str) -> bool:
        """Cancel an active or queued request, immediately reclaiming its KV resources."""
        # 1. Check waiting queue
        for i, req in enumerate(self._waiting):
            if req.request_id == request_id:
                req.mark_cancelled()
                self._waiting.pop(i)
                self._cancelled.append(req)
                return True

        # 2. Check running requests
        if request_id in self._running:
            req = self._running.pop(request_id)
            if not req.is_finished:
                req.mark_cancelled()
            self.kv_cache_mgr.release_request(request_id)
            self._cancelled.append(req)
            return True

        return False

    def fail_request(self, request_id: str, error: str) -> bool:
        """Mark a request as failed and immediately reclaim its KV resources."""
        if request_id in self._running:
            req = self._running.pop(request_id)
            if not req.is_finished:
                req.mark_failed(error)
            self.kv_cache_mgr.release_request(request_id)
            self._failed.append(req)
            return True

        for i, req in enumerate(self._waiting):
            if req.request_id == request_id:
                req.mark_failed(error)
                self._waiting.pop(i)
                self._failed.append(req)
                return True

        return False

    # --- Continuous Batching Scheduling Step ---

    def schedule(self) -> SchedulerBatch:
        """Construct the next iteration's batch of work.
        
        Pipeline:
        1. Drain completed / cancelled / failed requests from running set and release memory.
        2. Schedule active decode requests (prioritized to prevent decode starvation).
        3. Continue chunked prefill for running requests.
        4. Consult memory manager and admit waiting requests under sequence and token budgets.
        """
        # 1. Drain finished requests from running set
        finished_ids = [
            req_id for req_id, req in self._running.items() if req.is_finished
        ]
        for req_id in finished_ids:
            req = self._running.pop(req_id)
            self.kv_cache_mgr.release_request(req_id)
            if req.state == RequestState.FINISHED:
                self._completed.append(req)
            elif req.state == RequestState.CANCELLED:
                self._cancelled.append(req)
            elif req.state == RequestState.FAILED:
                self._failed.append(req)

        scheduled_items: list[ScheduledItem] = []
        batch_token_budget = self.max_batch_tokens
        prefill_token_budget = min(self.max_prefill_tokens, self.max_batch_tokens)

        # 2. Schedule active DECODE requests
        running_decodes = [
            req for req in self._running.values() if req.state == RequestState.DECODING
        ]
        sorted_decodes = self.policy.sort_running_decode(running_decodes)

        for req in sorted_decodes:
            if batch_token_budget < 1:
                break

            # Verify memory capacity for next decode token
            if self.kv_cache_mgr.can_append_token(req.request_id, req.total_tokens + 1):
                scheduled_items.append(
                    ScheduledItem(
                        request_id=req.request_id,
                        work_type=WorkType.DECODE,
                        num_tokens=1,
                        decode_step_idx=req.num_generated_tokens,
                    )
                )
                batch_token_budget -= 1

        # 3. Continue in-progress chunked PREFILL requests already in running set
        running_prefills = [
            req for req in self._running.values() if req.state == RequestState.PREFILL
        ]
        for req in running_prefills:
            if batch_token_budget <= 0 or prefill_token_budget <= 0:
                break

            remaining_prompt = req.num_prompt_tokens - req.num_prompt_tokens_processed
            chunk_size = min(
                remaining_prompt,
                self.prefill_chunk_size,
                batch_token_budget,
                prefill_token_budget,
            )
            if chunk_size > 0:
                start_idx = req.num_prompt_tokens_processed
                end_idx = start_idx + chunk_size
                scheduled_items.append(
                    ScheduledItem(
                        request_id=req.request_id,
                        work_type=WorkType.PREFILL,
                        num_tokens=chunk_size,
                        prompt_token_range=(start_idx, end_idx),
                    )
                )
                batch_token_budget -= chunk_size
                prefill_token_budget -= chunk_size

        # 4. Admit new WAITING requests (Continuous Batching)
        sorted_waiting = self.policy.sort_waiting(self._waiting)
        admitted_indices = []

        for idx, candidate in enumerate(sorted_waiting):
            # Check concurrency ceiling
            if len(self._running) >= self.max_num_sequences:
                break

            # Check token budget headroom
            if batch_token_budget <= 0 or prefill_token_budget <= 0:
                break

            # Check KV cache memory availability
            if not self.kv_cache_mgr.can_allocate_prompt_tokens(candidate.prompt_token_ids):
                # Temporary memory exhaustion: request safely waits; stop to preserve FCFS order
                break

            # Allocate physical blocks for candidate
            self.kv_cache_mgr.allocate_for_prompt_tokens(
                candidate.request_id, candidate.prompt_token_ids
            )
            candidate.mark_prefilling()
            self._running[candidate.request_id] = candidate
            admitted_indices.append(idx)

            # Schedule initial prefill chunk
            chunk_size = min(
                candidate.num_prompt_tokens,
                self.prefill_chunk_size,
                batch_token_budget,
                prefill_token_budget,
            )
            scheduled_items.append(
                ScheduledItem(
                    request_id=candidate.request_id,
                    work_type=WorkType.PREFILL,
                    num_tokens=chunk_size,
                    prompt_token_range=(0, chunk_size),
                )
            )
            batch_token_budget -= chunk_size
            prefill_token_budget -= chunk_size

        # Remove admitted requests from waiting queue (in reverse order to preserve indices)
        for idx in sorted(admitted_indices, reverse=True):
            admitted_req = sorted_waiting[idx]
            self._waiting.remove(admitted_req)

        return SchedulerBatch(items=scheduled_items)
