"""PagedAttention kernel execution backends."""

from pagedserve.kernels.backend import PagedAttentionBackend
from pagedserve.kernels.pytorch_backend import PyTorchPagedAttentionBackend
from pagedserve.kernels.metal_backend import MetalPagedAttentionBackend, is_metal_available


def get_backend(name: str = "auto") -> PagedAttentionBackend:
    """Backend factory function returning the requested PagedAttention backend.
    
    Args:
        name: 'auto', 'pytorch', or 'metal'.
              'auto': returns MetalPagedAttentionBackend if available, else PyTorchPagedAttentionBackend.
              
    Returns:
        Instance of PagedAttentionBackend.
    """
    normalized = name.lower().strip()
    if normalized == "pytorch":
        return PyTorchPagedAttentionBackend()
    elif normalized == "metal":
        return MetalPagedAttentionBackend()
    elif normalized == "auto":
        if is_metal_available():
            return MetalPagedAttentionBackend()
        return PyTorchPagedAttentionBackend()
    else:
        raise ValueError(f"Unknown backend name '{name}'. Expected 'auto', 'pytorch', or 'metal'.")
