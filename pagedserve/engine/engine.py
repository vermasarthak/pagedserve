"""Main PagedServe Inference Engine orchestrating memory, scheduler, runner, and sampler."""

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch

from pagedserve.config import EngineConfig
from pagedserve.engine.request import InferenceRequest, SamplingParams
from pagedserve.engine.state import RequestState
from pagedserve.memory.kv_cache import KVCacheManager
from pagedserve.model.loader import ModelLoader, LoadedModel
from pagedserve.model.runner import ModelRunner, ModelSequenceState
from pagedserve.model.sampler import Sampler
from pagedserve.scheduler.batch import WorkType, SchedulerBatch
from pagedserve.scheduler.scheduler import Scheduler
from pagedserve.errors import InvalidRequestError, PagedServeError


@dataclass
class EngineStepOutput:
    """Output produced by a single engine execution step."""

    batch: SchedulerBatch
    generated_tokens: Dict[str, int] = field(default_factory=dict)
    finished_requests: List[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return self.batch.is_empty and len(self.generated_tokens) == 0


class PagedServeEngine:
    """The central orchestrator of the PagedServe LLM serving runtime.
    
    Coordinates:
    - Request submission & tokenization
    - KV Cache block management (KVCacheManager)
    - Continuous batching scheduling (Scheduler)
    - PyTorch model execution (ModelRunner)
    - Token sampling (Sampler)
    - Safe resource cleanup on completion, cancellation, or failure.
    """

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        loaded_model: Optional[LoadedModel] = None,
    ):
        self.config: EngineConfig = config if config is not None else EngineConfig()

        # 1. Load Model & Tokenizer
        if loaded_model is not None:
            self.loaded_model: LoadedModel = loaded_model
        else:
            self.loaded_model = ModelLoader.load(
                model_name_or_path=self.config.model_name_or_path,
                device=self.config.device,
                dtype=self.config.dtype,
            )

        self.tokenizer = self.loaded_model.tokenizer
        self.device = self.loaded_model.device

        # 2. Initialize Memory Manager
        self.kv_cache_mgr: KVCacheManager = KVCacheManager(
            num_blocks=self.config.num_blocks,
            block_size=self.config.block_size,
            enable_prefix_caching=self.config.enable_prefix_caching,
        )

        # 3. Initialize Continuous Batching Scheduler
        self.scheduler: Scheduler = Scheduler(
            kv_cache_mgr=self.kv_cache_mgr,
            max_num_sequences=self.config.max_num_sequences,
            max_batch_tokens=self.config.max_batch_tokens,
            max_prefill_tokens=self.config.max_prefill_tokens_per_step,
            prefill_chunk_size=self.config.max_prefill_tokens_per_step,
        )

        # 4. Initialize Model Runner & Sampler
        self.runner: ModelRunner = ModelRunner(loaded_model=self.loaded_model)
        self.sampler: Sampler = Sampler()

        # 5. Engine State Tracking
        self._requests: Dict[str, InferenceRequest] = {}
        self._model_states: Dict[str, ModelSequenceState] = {}

    @property
    def has_active_work(self) -> bool:
        """True if any requests are queued or currently executing."""
        return self.scheduler.num_waiting > 0 or self.scheduler.num_running > 0

    def submit(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
    ) -> str:
        """Tokenize a prompt and enqueue an inference request for continuous scheduling.
        
        Args:
            prompt: Text prompt string.
            sampling_params: Sampling configuration (defaults to greedy/temp=1.0).
            request_id: Optional user-supplied unique ID.
            
        Returns:
            The request_id assigned to this request.
        """
        if not prompt:
            raise InvalidRequestError("Prompt cannot be empty")

        req_id = request_id if request_id is not None else str(uuid.uuid4())
        if req_id in self._requests:
            raise InvalidRequestError(f"Request ID '{req_id}' is already registered")

        # Tokenize prompt
        prompt_token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not prompt_token_ids:
            # Fallback for whitespace or empty tokenization
            prompt_token_ids = [self.tokenizer.eos_token_id or 0]

        # Configure sampling params with tokenizer's EOS token if not explicitly overridden
        params = sampling_params if sampling_params is not None else SamplingParams()
        if self.tokenizer.eos_token_id is not None:
            params.stop_token_ids.add(self.tokenizer.eos_token_id)

        req = InferenceRequest(
            request_id=req_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            sampling_params=params,
        )

        self._requests[req_id] = req
        self.scheduler.add_request(req)
        return req_id

    def cancel(self, request_id: str) -> bool:
        """Cancel an active or pending request and immediately reclaim all resources."""
        # 1. Clean up scheduler and metadata KV blocks
        cancelled = self.scheduler.cancel_request(request_id)

        # 2. Clean up actual PyTorch model KV state
        self._model_states.pop(request_id, None)

        return cancelled

    def get_request(self, request_id: str) -> Optional[InferenceRequest]:
        """Retrieve request metadata and execution state."""
        return self._requests.get(request_id)

    def get_output_text(self, request_id: str) -> str:
        """Decode the generated token IDs into a text string."""
        req = self._requests.get(request_id)
        if req is None or not req.generated_token_ids:
            return ""
        return self.tokenizer.decode(req.generated_token_ids, skip_special_tokens=True)

    def step(self) -> EngineStepOutput:
        """Execute a single engine iteration step.
        
        1. Queries Scheduler for the next batch of work (PREFILL and DECODE slices).
        2. Executes prefill forward passes on prompt chunks; samples initial token when prompt completes.
        3. Executes single-token decode forward passes on active decoding sequences; samples next tokens.
        4. Updates sequence state machines and releases completed resources.
        """
        batch = self.scheduler.schedule()
        output = EngineStepOutput(batch=batch)

        if batch.is_empty:
            self._cleanup_finished_states()
            return output

        # --- Phase 1: Execute PREFILL items ---
        for item in batch.prefill_items:
            req = self.scheduler.running_requests.get(item.request_id)
            if req is None:
                continue

            try:
                start_idx, end_idx = item.prompt_token_range
                tokens_to_process = req.prompt_token_ids[start_idx:end_idx]

                existing_state = self._model_states.get(item.request_id)
                logits, new_state = self.runner.prefill(
                    request_id=item.request_id,
                    token_ids=tokens_to_process,
                    existing_state=existing_state,
                )
                self._model_states[item.request_id] = new_state
                req.advance_prompt_tokens(item.num_tokens)

                # If prompt is now fully ingested, transition to DECODING and sample the first token!
                if req.is_prefill_complete:
                    req.mark_decoding()
                    first_token = self.sampler.sample(logits, req.sampling_params)
                    req.append_generated_token(first_token)
                    output.generated_tokens[item.request_id] = first_token

                    # Record token in memory manager
                    self.kv_cache_mgr.append_token(req.request_id, req.total_tokens)

            except Exception as e:
                self.scheduler.fail_request(item.request_id, str(e))
                self._model_states.pop(item.request_id, None)

        # --- Phase 2: Execute DECODE items ---
        for item in batch.decode_items:
            req = self.scheduler.running_requests.get(item.request_id)
            state = self._model_states.get(item.request_id)
            if req is None or state is None:
                continue

            try:
                # The input to decode is the latest generated token
                last_token_id = req.generated_token_ids[-1]

                logits, updated_state = self.runner.decode(
                    request_id=item.request_id,
                    next_token_id=last_token_id,
                    state=state,
                )
                self._model_states[item.request_id] = updated_state

                # Sample next token
                next_token = self.sampler.sample(logits, req.sampling_params)
                req.append_generated_token(next_token)
                output.generated_tokens[item.request_id] = next_token

                # Record token expansion in memory manager
                self.kv_cache_mgr.append_token(req.request_id, req.total_tokens)

            except Exception as e:
                self.scheduler.fail_request(item.request_id, str(e))
                self._model_states.pop(item.request_id, None)

        # --- Phase 3: Cleanup Terminal Model States ---
        self._cleanup_finished_states()

        # Track any requests that finished in this step
        for req_id, token_id in output.generated_tokens.items():
            req = self._requests.get(req_id)
            if req and req.is_finished:
                output.finished_requests.append(req_id)

        return output

    def _cleanup_finished_states(self) -> None:
        """Purge model KV activation states for finished, cancelled, or failed requests."""
        terminal_ids = [
            req_id for req_id in self._model_states.keys()
            if req_id not in self.scheduler.running_requests
            or self._requests[req_id].is_finished
        ]
        for req_id in terminal_ids:
            self._model_states.pop(req_id, None)

    def run_until_complete(self, request_id: str, max_steps: int = 1000) -> InferenceRequest:
        """Convenience driver that repeatedly steps the engine until a request terminates."""
        req = self._requests.get(request_id)
        if req is None:
            raise KeyError(f"Unknown request ID '{request_id}'")

        steps = 0
        while not req.is_finished and steps < max_steps:
            self.step()
            steps += 1

        if not req.is_finished:
            raise RuntimeError(f"Request '{request_id}' exceeded max execution steps ({max_steps})")

        return req
