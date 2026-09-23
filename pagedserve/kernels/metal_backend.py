"""Custom fused Apple Metal execution backend for single-token decode paged attention."""

import ctypes
import ctypes.util
import math
import os

import torch

from pagedserve.kernels.backend import PagedAttentionBackend
from pagedserve.kernels.pytorch_backend import PyTorchPagedAttentionBackend
from pagedserve.memory.paged_view import PagedKVView

# Objective-C / Metal framework initialization
_objc = None
_metal = None
_foundation = None
_metal_device = None
_metal_available = False

try:
    _objc_path = ctypes.util.find_library("objc")
    if _objc_path is not None:
        _objc = ctypes.cdll.LoadLibrary(_objc_path)
        _metal = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Metal.framework/Metal")
        _foundation = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Foundation.framework/Foundation")

    _objc.objc_getClass.restype = ctypes.c_void_p
    _objc.objc_getClass.argtypes = [ctypes.c_char_p]

    _objc.sel_registerName.restype = ctypes.c_void_p
    _objc.sel_registerName.argtypes = [ctypes.c_char_p]

    _metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
    _metal.MTLCreateSystemDefaultDevice.argtypes = []

    _metal_device = _metal.MTLCreateSystemDefaultDevice()
    if _metal_device:
        _metal_available = True
except Exception:
    _metal_available = False


def is_metal_available() -> bool:
    """Return True if custom Metal device kernel execution is supported in this environment."""
    return _metal_available


def _get_sel(name: str):
    return _objc.sel_registerName(name.encode("utf-8"))


def _get_class(name: str):
    return _objc.objc_getClass(name.encode("utf-8"))


def _msg(obj, sel_name: str, restype, argtypes, *args):
    func = _objc.objc_msgSend
    func.restype = restype
    func.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + argtypes
    return func(obj, _get_sel(sel_name), *args)


def _ns_str(s: str):
    NSString = _get_class("NSString")
    return _msg(NSString, "stringWithUTF8String:", ctypes.c_void_p, [ctypes.c_char_p], s.encode("utf-8"))


class MTLSize(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_ulong),
        ("height", ctypes.c_ulong),
        ("depth", ctypes.c_ulong),
    ]


class MetalPagedAttentionBackend(PagedAttentionBackend):
    """Fused Apple Metal execution backend for single-token decode paged attention.

    Reads physical KV blocks directly from TensorBlockStore via logical block tables.
    Iterates through non-contiguous pages directly in Metal Shading Language (MSL)
    with online numerically stable softmax.
    """

    def __init__(self):
        self._fallback = PyTorchPagedAttentionBackend()
        self._pipeline_cache: dict[str, ctypes.c_void_p] = {}
        self._cmd_queue = None

        if _metal_available and _metal_device:
            self._cmd_queue = _msg(_metal_device, "newCommandQueue", ctypes.c_void_p, [])

    def _get_pipeline(self) -> ctypes.c_void_p | None:
        """Compile and cache the MSL compute shader pipeline."""
        if not _metal_available or not _metal_device:
            return None

        cache_key = "paged_attention_decode_kernel"
        if cache_key in self._pipeline_cache:
            return self._pipeline_cache[cache_key]

        shader_path = os.path.join(os.path.dirname(__file__), "metal_shader.metal")
        if not os.path.exists(shader_path):
            return None

        with open(shader_path, "r", encoding="utf-8") as f:
            shader_source = f.read()

        err = ctypes.c_void_p(0)
        lib = _msg(
            _metal_device,
            "newLibraryWithSource:options:error:",
            ctypes.c_void_p,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)],
            _ns_str(shader_source),
            None,
            ctypes.byref(err),
        )
        if not lib:
            return None

        fn = _msg(lib, "newFunctionWithName:", ctypes.c_void_p, [ctypes.c_void_p], _ns_str(cache_key))
        if not fn:
            return None

        pipeline = _msg(
            _metal_device,
            "newComputePipelineStateWithFunction:error:",
            ctypes.c_void_p,
            [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)],
            fn,
            ctypes.byref(err),
        )
        if not pipeline:
            return None

        self._pipeline_cache[cache_key] = pipeline
        return pipeline

    def forward(
        self,
        query: torch.Tensor,
        paged_view: PagedKVView,
        layer_idx: int,
        scale: float | None = None,
    ) -> torch.Tensor:
        """Execute single-token decode paged attention on GPU using custom Metal kernel.

        Falls back to PyTorchPagedAttentionBackend if shape or environment is unsupported.
        """
        # Guard: Metal backend targets single-token decode (query_seq_len == 1)
        query_seq_len = query.shape[2]
        if query_seq_len != 1 or not _metal_available:
            return self._fallback.forward(query, paged_view, layer_idx, scale)

        pipeline = self._get_pipeline()
        if not pipeline or not self._cmd_queue:
            return self._fallback.forward(query, paged_view, layer_idx, scale)

        # Extract dimensions
        num_attn_heads = query.shape[1]
        head_dim = query.shape[3]
        store = paged_view.store
        geom = store.geometry
        num_kv_heads = geom.num_kv_heads
        block_size = geom.block_size
        total_blocks = store.num_blocks
        seq_length = paged_view.seq_length
        block_table = paged_view.block_table.blocks
        num_blocks_in_table = len(block_table)

        if head_dim > 128:  # Metal register array limit in shader
            return self._fallback.forward(query, paged_view, layer_idx, scale)

        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)

        # Convert tensors to contiguous float32 CPU byte buffers for Metal buffer wrapping
        query_flat = query.detach().to(device="cpu", dtype=torch.float32).contiguous()
        k_store_flat = store._k_store.detach().to(device="cpu", dtype=torch.float32).contiguous()
        v_store_flat = store._v_store.detach().to(device="cpu", dtype=torch.float32).contiguous()
        block_table_tensor = torch.tensor(block_table, dtype=torch.int32, device="cpu").contiguous()

        q_bytes = query_flat.numpy().tobytes()
        k_bytes = k_store_flat.numpy().tobytes()
        v_bytes = v_store_flat.numpy().tobytes()
        bt_bytes = block_table_tensor.numpy().tobytes()

        # MTLResourceStorageModeShared = 0
        q_buf = _msg(_metal_device, "newBufferWithBytes:length:options:", ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong], q_bytes, len(q_bytes), 0)
        k_buf = _msg(_metal_device, "newBufferWithBytes:length:options:", ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong], k_bytes, len(k_bytes), 0)
        v_buf = _msg(_metal_device, "newBufferWithBytes:length:options:", ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong], v_bytes, len(v_bytes), 0)
        bt_buf = _msg(_metal_device, "newBufferWithBytes:length:options:", ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong], bt_bytes, len(bt_bytes), 0)

        out_len_bytes = num_attn_heads * head_dim * 4
        out_buf = _msg(_metal_device, "newBufferWithLength:options:", ctypes.c_void_p, [ctypes.c_ulong, ctypes.c_ulong], out_len_bytes, 0)

        # Encode command buffer
        cmd_buf = _msg(self._cmd_queue, "commandBuffer", ctypes.c_void_p, [])
        encoder = _msg(cmd_buf, "computeCommandEncoder", ctypes.c_void_p, [])

        _msg(encoder, "setComputePipelineState:", None, [ctypes.c_void_p], pipeline)
        _msg(encoder, "setBuffer:offset:atIndex:", None, [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong], q_buf, 0, 0)
        _msg(encoder, "setBuffer:offset:atIndex:", None, [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong], k_buf, 0, 1)
        _msg(encoder, "setBuffer:offset:atIndex:", None, [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong], v_buf, 0, 2)
        _msg(encoder, "setBuffer:offset:atIndex:", None, [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong], bt_buf, 0, 3)
        _msg(encoder, "setBuffer:offset:atIndex:", None, [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong], out_buf, 0, 4)

        # Set scalar constants via setBytes:length:atIndex:
        def set_int(val: int, idx: int):
            c_val = ctypes.c_int(val)
            _msg(encoder, "setBytes:length:atIndex:", None, [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong], ctypes.byref(c_val), 4, idx)

        def set_float(val: float, idx: int):
            c_val = ctypes.c_float(val)
            _msg(encoder, "setBytes:length:atIndex:", None, [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong], ctypes.byref(c_val), 4, idx)

        set_int(num_attn_heads, 5)
        set_int(num_kv_heads, 6)
        set_int(head_dim, 7)
        set_int(block_size, 8)
        set_int(total_blocks, 9)
        set_int(num_blocks_in_table, 10)
        set_int(seq_length, 11)
        set_int(layer_idx, 12)
        set_float(scale, 13)

        # Dispatch grid size: 1 thread per query head
        grid_size = MTLSize(num_attn_heads, 1, 1)
        threadgroup_size = MTLSize(min(num_attn_heads, 32), 1, 1)

        msg_dispatch = _objc.objc_msgSend
        msg_dispatch.restype = None
        msg_dispatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, MTLSize, MTLSize]
        msg_dispatch(encoder, _get_sel("dispatchThreads:threadsPerThreadgroup:"), grid_size, threadgroup_size)

        _msg(encoder, "endEncoding", None, [])
        _msg(cmd_buf, "commit", None, [])
        _msg(cmd_buf, "waitUntilCompleted", None, [])

        # Read output from Metal buffer
        contents_ptr = _msg(out_buf, "contents", ctypes.c_void_p, [])
        out_bytes = ctypes.string_at(contents_ptr, out_len_bytes)
        out_tensor = torch.frombuffer(bytearray(out_bytes), dtype=torch.float32).reshape(1, num_attn_heads, 1, head_dim)

        return out_tensor.to(device=query.device, dtype=query.dtype)
