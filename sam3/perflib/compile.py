# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

from collections.abc import Mapping
from contextlib import ExitStack, contextmanager, nullcontext
from functools import wraps
from typing import Any

import torch
from sam3.model.data_misc import BatchedDatapoint, NestedTensor
from torch._dynamo import config as dynamo_config
from torch.utils._pytree import tree_map_only


# >>> CHANGE: scope SAM3 Dynamo knobs to compiled-call boundaries. <<<
SAM3_COMPILE_DYNAMO_CONFIG: dict[str, Any] = {
    "cache_size_limit": 128,
    "accumulated_cache_size_limit": 2048,
    "capture_scalar_outputs": True,
}

# >>> CHANGE: preserve tracker-only cache sizing without process-global writes. <<<
SAM3_TRACKER_COMPILE_DYNAMO_CONFIG: dict[str, Any] = {
    "cache_size_limit": 64,
    "accumulated_cache_size_limit": 2048,
}

# >>> CHANGE: keep activation-checkpoint Dynamo-DDP tweaks local to compiled calls. <<<
SAM3_ACT_CKPT_DYNAMO_CONFIG: dict[str, Any] = {
    "optimize_ddp": False,
}


# >>> CHANGE: do not set torch._dynamo.config globally during lazy torch.compile. <<<
def dynamo_config_context(config: Mapping[str, Any] | None = None):
    """Temporarily apply Dynamo config while a compiled callable executes."""
    if config is None:
        return nullcontext()
    return dynamo_config.patch(dict(config))


# >>> CHANGE: reusable wrapper for SAM3 compiled call sites. <<<
def wrap_with_dynamo_config(f, *, config: Mapping[str, Any] | None = None):
    """Run one callable under a temporary Dynamo config patch."""

    @wraps(f)
    def wrapped(*args, **kwargs):
        with dynamo_config_context(config):
            return f(*args, **kwargs)

    return wrapped


# >>> CHANGE: compile and scope Dynamo config at execution time, not construction time. <<<
def compile_with_dynamo_config(fn, *, config: Mapping[str, Any] | None = None, **compile_kwargs):
    """Build a lazy torch.compile callable whose execution patches Dynamo locally."""
    compiled_fn = torch.compile(fn, **compile_kwargs)
    return wrap_with_dynamo_config(compiled_fn, config=config)


# >>> CHANGE: scope SAM3 backend precision flags instead of mutating process globals. <<<
@contextmanager
def backend_config_context(*, allow_tf32: bool | None = None,
                           cudnn_benchmark: bool | None = None, cudnn_deterministic: bool | None = None):
    """Temporarily apply process-global CUDA/cuDNN backend flags."""
    matmul_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32
    benchmark = torch.backends.cudnn.benchmark
    deterministic = torch.backends.cudnn.deterministic
    try:
        if allow_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
        if cudnn_benchmark is not None:
            torch.backends.cudnn.benchmark = cudnn_benchmark
        if cudnn_deterministic is not None:
            torch.backends.cudnn.deterministic = cudnn_deterministic
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_allow_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_allow_tf32
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic


# >>> CHANGE: reusable SAM3 inference precision context for entry-point wrappers. <<<
def sam3_backend_context(*, allow_tf32: bool = True):
    """Apply SAM3's original TF32 preference only inside one explicit scope."""
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        if device_props.major >= 8:
            return backend_config_context(allow_tf32=allow_tf32)
    return nullcontext()


# >>> CHANGE: combine SAM3 backend and autocast preferences at explicit entry points. <<<
@contextmanager
def sam3_inference_context(*, allow_tf32: bool = True,
                           autocast_dtype: torch.dtype | None = torch.bfloat16, device_type: str = "cuda"):
    """Apply SAM3 inference precision preferences for one bounded call."""
    with ExitStack() as stack:
        stack.enter_context(sam3_backend_context(allow_tf32=allow_tf32))
        if autocast_dtype is not None and torch.cuda.is_available():
            stack.enter_context(
                torch.autocast(device_type=device_type, dtype=autocast_dtype)
            )
        yield


def recursive_fn_factory(fn):
    def recursive_fn(b):
        if isinstance(b, dict):
            return {k: recursive_fn(b[k]) for k in b}
        if isinstance(b, list):
            return [recursive_fn(t) for t in b]
        if isinstance(b, tuple):
            return tuple(recursive_fn(t) for t in b)
        if isinstance(b, NestedTensor):
            tensors = fn(b.tensors)
            if b.mask is None:
                mask = None
            else:
                mask = fn(b.mask)
            return NestedTensor(tensors=tensors, mask=mask)
        if isinstance(b, torch.Tensor):
            return fn(b)
        if b is None:
            return b
        trivial_types = [bool, int, float]
        for t in trivial_types:
            if isinstance(b, t):
                return b
        raise TypeError(f"Unexpected type {type(b)}")

    return recursive_fn


recursive_contiguous = recursive_fn_factory(lambda x: x.contiguous())
recursive_clone = recursive_fn_factory(torch.clone)


def clone_output_wrapper(f):
    """
    Clone the CUDA output tensors of a function to avoid in-place operations.
    Uses tree_map_only (C-optimized pytree traversal) matching onevision's pattern.
    Requires NestedTensor to be registered as a pytree node (see data_misc.py).
    """

    @wraps(f)
    def wrapped(*args, **kwargs):
        outputs = f(*args, **kwargs)
        return tree_map_only(
            torch.Tensor, lambda t: t.clone() if t.is_cuda else t, outputs
        )

    return wrapped


def compile_wrapper(
    fn, *, mode="max-autotune", fullgraph=True, dynamic=False, name=None, config: Mapping[str, Any] | None = None,
):
    """Compile with recursive_contiguous on inputs and recursive_clone on outputs.
    Used for SAM2 tracker components that need contiguous inputs for CUDA graphs."""
    compiled_fn = torch.compile(fn, mode=mode, fullgraph=fullgraph, dynamic=dynamic)

    def compiled_fn_wrapper(*args, **kwargs):
        with torch.autograd.profiler.record_function(
            f"compiled {fn}" if name is None else name
        ):
            CUDAGRAPH_MODES = ["max-autotune", "reduce-overhead"]
            args = recursive_contiguous(args)
            kwargs = recursive_contiguous(kwargs)
            # >>> CHANGE: lazy Dynamo tracing/recompiles must see only local config. <<<
            # Original SAM3 implementation:
            # result = compiled_fn(*args, **kwargs)
            with dynamo_config_context(config):
                result = compiled_fn(*args, **kwargs)
            if mode in CUDAGRAPH_MODES:
                result = recursive_clone(result)
            return result

    return compiled_fn_wrapper


def shape_logging_wrapper(fn, keep_kwargs, enable_logging=False):
    """
    Wraps a function and prints the shapes of all tensor inputs.
    Only prints when a new combination of shapes is seen.
    """
    seen_shapes = set()

    def get_shape(obj):
        if isinstance(obj, torch.Tensor):
            return obj.shape
        elif isinstance(obj, (list, tuple)):
            if len(obj) > 1:
                return tuple(get_shape(x) for x in obj)
            return get_shape(obj[0])
        elif isinstance(obj, dict):
            return tuple(sorted((k, get_shape(v)) for k, v in obj.items()))
        else:
            return type(obj).__name__

    def wrapper(*args, **kwargs):
        shapes = tuple(get_shape(arg) for arg in args) + tuple(
            (k, get_shape(v))
            for k, v in kwargs.items()
            if isinstance(v, (torch.Tensor, list))
            and (len(keep_kwargs) > 0 and k in keep_kwargs)
        )
        if shapes not in seen_shapes:
            seen_shapes.add(shapes)
            if enable_logging:
                print(f"[ShapeLogger] New input shapes for {fn.__qualname__}: {shapes}")
        return fn(*args, **kwargs)

    wrapper.enable_logging = enable_logging

    def set_logging(enabled=False):
        nonlocal enable_logging
        enable_logging = enabled
        wrapper.enable_logging = enable_logging

    wrapper.set_logging = set_logging
    return wrapper
