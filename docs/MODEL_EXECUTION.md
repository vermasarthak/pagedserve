# Model Execution

The Model Execution subsystem in PagedServe is responsible for coordinating the actual PyTorch transformer models to perform inference. It interfaces with the continuous batching scheduler and the KV cache manager to orchestrate prefill and decode steps.

## Components

### ModelLoader
The `ModelLoader` is responsible for loading the Hugging Face transformer model, resolving the optimal target device (CUDA, MPS, CPU) and data type, and extracting necessary architectural metadata (like hidden sizes, number of layers, number of heads).

### ModelRunner
The `ModelRunner` orchestrates the forward passes of the model. It handles:
- **Prefill**: Processing the prompt tokens, supporting chunked prefill to respect maximum token budgets.
- **Decode**: Processing a single token at a time for all requests in a batch in a single autoregressive step.
It utilizes Hugging Face's `DynamicCache` to retain Key/Value states during execution, working within a `torch.inference_mode()` context.

### Sampler
The `Sampler` processes the raw logits returned by the model and selects the next token. The sampling pipeline is:
1. Raw Logits
2. Temperature Scaling
3. Top-K Filtering
4. Top-P (Nucleus) Filtering
5. Softmax
6. Multinomial Sampling or Greedy (Argmax)

### Engine Lifecycle
The `PagedServeEngine` is the central orchestrator. It runs a loop that:
1. Steps the scheduler to form a batch of prefill and decode work.
2. Passes the batch to the `ModelRunner`.
3. Passes the logits to the `Sampler`.
4. Updates request states with the new tokens.
5. Reclaims resources for finished or cancelled requests.

## Metadata KV Blocks vs Physical Transformer KV Tensors

### Architecture Evolution
- **BEFORE (Phases 1–20)**: Metadata-only block paging + Hugging Face `DynamicCache`.
- **CURRENT (Phases 21–28)**: Metadata paging + Real block-backed physical KV tensor pool (`TensorBlockStore`) + Reference scatter/gather & paged attention path (`PagedKVView` / `paged_attention_reference`).

> [!NOTE]
> **Production Generation Path Notice**  
> While PagedServe now possesses a fully verified physical tensor store, scatter/gather engine, and reference paged attention, the primary production continuous batching generation path (`ModelRunner`) currently continues to use Hugging Face's native `DynamicCache` for model forward passes. This maintains maximum compatibility while establishing exact numerical correctness for the physical block store before optimized physical CUDA/Triton kernels are introduced.
