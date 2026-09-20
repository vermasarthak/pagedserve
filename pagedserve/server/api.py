"""PagedServe FastAPI HTTP Server."""

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from pagedserve.config import EngineConfig
from pagedserve.engine.engine import PagedServeEngine
from pagedserve.engine.request import SamplingParams
from pagedserve.memory.kv_cache import KVCacheManager
from pagedserve.model.loader import ModelLoader
from pagedserve.scheduler.policy import FCFSPolicy
from pagedserve.scheduler.scheduler import Scheduler
from pagedserve.server.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    UsageInfo,
)
from pagedserve.server.streaming import (
    format_sse_done,
    format_sse_event,
    stream_event_to_chat_chunk,
    stream_event_to_completion_chunk,
)


def format_chat_messages(messages: list[ChatMessage]) -> str:
    """Convert chat messages to text using a simple deterministic fallback."""
    parts = []
    for msg in messages:
        role = msg.role.capitalize()
        parts.append(f"{role}: {msg.content}")
    parts.append("Assistant:")
    return "\n".join(parts)


_engine: PagedServeEngine | None = None
_config: EngineConfig | None = None


def get_engine() -> PagedServeEngine:
    if _engine is None:
        raise RuntimeError("Engine not initialized. Start server with create_app() or lifespan.")
    return _engine


def create_app(
    model_name_or_path: str = "sshleifer/tiny-gpt2",
    device: str | None = None,
    dtype: str | None = None,
    num_blocks: int = 256,
    block_size: int = 16,
    max_num_sequences: int = 16,
    max_batch_tokens: int = 2048,
    max_prefill_tokens_per_step: int = 512,
    enable_prefix_caching: bool = True,
    engine_instance: PagedServeEngine | None = None,
) -> FastAPI:
    """Factory that creates a FastAPI application with a shared PagedServe engine."""
    global _engine, _config

    cfg = EngineConfig(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=dtype or "float32",
        num_blocks=num_blocks,
        block_size=block_size,
        max_num_sequences=max_num_sequences,
        max_batch_tokens=max_batch_tokens,
        max_prefill_tokens_per_step=max_prefill_tokens_per_step,
        enable_prefix_caching=enable_prefix_caching,
    )
    _config = cfg

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _engine
        if engine_instance is not None:
            _engine = engine_instance
        else:
            loaded = ModelLoader.load(
                model_name_or_path=cfg.model_name_or_path,
                device=cfg.device,
                dtype=cfg.dtype,
            )
            kv_cache = KVCacheManager(
                block_size=cfg.block_size,
                num_blocks=cfg.num_blocks,
                enable_prefix_caching=cfg.enable_prefix_caching,
            )
            scheduler = Scheduler(
                kv_cache_mgr=kv_cache,
                max_num_sequences=cfg.max_num_sequences,
                max_batch_tokens=cfg.max_batch_tokens,
                max_prefill_tokens=cfg.max_prefill_tokens_per_step,
                prefill_chunk_size=cfg.max_prefill_tokens_per_step,
                policy=FCFSPolicy(),
            )
            _engine = PagedServeEngine(cfg, loaded, scheduler, kv_cache)
        yield
        _engine = None

    application = FastAPI(
        title="PagedServe",
        description="Experimental LLM Inference Runtime with Continuous Batching",
        version="0.1.0",
        lifespan=lifespan,
    )
    _attach_routes(application)
    return application


def _attach_routes(app: FastAPI) -> None:
    """Register all API routes on the given FastAPI app."""

    @app.get("/health")
    async def health():
        engine = get_engine()
        engine.update_metrics_snapshot()
        cfg = _config
        return {
            "status": "ok",
            "model": cfg.model_name_or_path if cfg else "unknown",
            "device": str(engine.loaded_model.device),
            "dtype": str(engine.loaded_model.dtype),
            "active_requests": engine.scheduler.num_running,
            "waiting_requests": engine.scheduler.num_waiting,
        }

    @app.get("/metrics")
    async def metrics():
        engine = get_engine()
        engine.update_metrics_snapshot()
        return engine.metrics.to_dict()

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest, http_request: Request):
        engine = get_engine()
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_new_tokens=request.max_tokens,
        )
        eos_id = engine.tokenizer.eos_token_id
        if eos_id is not None:
            sampling_params.stop_token_ids.add(eos_id)

        try:
            req_id = engine.submit(prompt=request.prompt, sampling_params=sampling_params)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        prompt_tokens = len(engine._requests[req_id].prompt_token_ids)

        if request.stream:
            return StreamingResponse(
                _stream_completion(engine, req_id, request.model),
                media_type="text/event-stream",
            )

        req = await asyncio.get_event_loop().run_in_executor(
            None, lambda: engine.run_until_complete(req_id)
        )
        text = engine.get_output_text(req_id)
        return CompletionResponse(
            id=req_id,
            model=request.model,
            choices=[CompletionChoice(text=text, finish_reason=req.finish_reason)],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=req.num_generated_tokens,
                total_tokens=prompt_tokens + req.num_generated_tokens,
            ),
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest, http_request: Request):
        engine = get_engine()

        tokenizer = engine.tokenizer
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None) is not None:
            try:
                prompt = tokenizer.apply_chat_template(
                    [{"role": m.role, "content": m.content} for m in request.messages],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                prompt = format_chat_messages(request.messages)
        else:
            prompt = format_chat_messages(request.messages)

        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_new_tokens=request.max_tokens,
        )
        eos_id = tokenizer.eos_token_id
        if eos_id is not None:
            sampling_params.stop_token_ids.add(eos_id)

        try:
            req_id = engine.submit(prompt=prompt, sampling_params=sampling_params)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        prompt_tokens = len(engine._requests[req_id].prompt_token_ids)

        if request.stream:
            return StreamingResponse(
                _stream_chat(engine, req_id, request.model),
                media_type="text/event-stream",
            )

        req = await asyncio.get_event_loop().run_in_executor(
            None, lambda: engine.run_until_complete(req_id)
        )
        text = engine.get_output_text(req_id)
        return ChatCompletionResponse(
            id=req_id,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason=req.finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=req.num_generated_tokens,
                total_tokens=prompt_tokens + req.num_generated_tokens,
            ),
        )


async def _stream_completion(engine: PagedServeEngine, req_id: str, model: str):
    loop = asyncio.get_event_loop()
    req = engine._requests[req_id]

    while not req.is_finished:
        await loop.run_in_executor(None, engine.step)
        q = engine._stream_queues.get(req_id)
        if q:
            while not q.empty():
                event = q.get_nowait()
                if event is None:
                    break
                chunk = stream_event_to_completion_chunk(event, model, req_id)
                yield format_sse_event(chunk)

    yield format_sse_done()


async def _stream_chat(engine: PagedServeEngine, req_id: str, model: str):
    loop = asyncio.get_event_loop()
    req = engine._requests[req_id]

    while not req.is_finished:
        await loop.run_in_executor(None, engine.step)
        q = engine._stream_queues.get(req_id)
        if q:
            while not q.empty():
                event = q.get_nowait()
                if event is None:
                    break
                chunk = stream_event_to_chat_chunk(event, model, req_id)
                yield format_sse_event(chunk)

    yield format_sse_done()


app = create_app(
    model_name_or_path=os.environ.get("PAGEDSERVE_MODEL", "sshleifer/tiny-gpt2"),
    device=os.environ.get("PAGEDSERVE_DEVICE", None),
    num_blocks=int(os.environ.get("PAGEDSERVE_NUM_BLOCKS", "256")),
)
