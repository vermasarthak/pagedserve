"""Core exceptions for PagedServe inference engine and memory subsystem."""


class PagedServeError(Exception):
    """Base exception for all PagedServe runtime errors."""
    pass


# Memory Subsystem Exceptions

class MemoryError(PagedServeError):
    """Base class for memory and block allocation errors."""
    pass


class KVCacheExhaustedError(MemoryError):
    """Raised when the BlockPool runs out of free blocks to satisfy an allocation request."""
    def __init__(self, requested: int, available: int, total: int):
        super().__init__(
            f"KV cache memory exhausted: requested {requested} blocks, "
            f"but only {available}/{total} blocks are available."
        )
        self.requested = requested
        self.available = available
        self.total = total


class InvalidBlockError(MemoryError):
    """Raised when an invalid or out-of-range physical block ID is referenced."""
    pass


class DoubleFreeError(MemoryError):
    """Raised when attempting to release an already-freed block or decrement ref_count below zero."""
    pass


class BlockInvariantError(MemoryError):
    """Raised when an internal invariant of a KVBlock or BlockPool is violated."""
    pass


# Request & Scheduling Exceptions

class RequestError(PagedServeError):
    """Base class for request validation and lifecycle errors."""
    pass


class InvalidRequestError(RequestError):
    """Raised when a request specification or sampling parameter is invalid."""
    pass


class InvalidRequestStateError(RequestError):
    """Raised when an illegal request state transition is attempted."""
    pass


class RequestCancelledError(RequestError):
    """Raised or recorded when a request is cancelled by the client or caller."""
    pass
