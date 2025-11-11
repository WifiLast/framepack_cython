# cython: language_level=3, boundscheck=False, wraparound=False

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple

import cython
import torch


cpu: torch.device = torch.device('cpu')
gpu: torch.device = torch.device(f'cuda:{torch.cuda.current_device()}')
gpu_complete_modules: List[torch.nn.Module] = []

cdef int _MAX_MEM_CACHE = 16
cdef int _mem_cache_size = 0
cdef int _mem_cache_devices[16]
cdef double _mem_cache_timestamp[16]
cdef double _mem_cache_value[16]


@dataclass
class MemoryOptimizationConfig:
    use_async_streams: bool = True
    use_pinned_memory: bool = True
    cache_memory_stats: bool = True
    stats_cache_ttl: float = 0.05

    def enable_async_copy(self) -> bool:
        return self.use_async_streams and torch.cuda.is_available()


@cython.locals(device=cython.object)
def _device_index(device: cython.object) -> int:
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, torch.device):
        if device.type != "cuda":
            raise ValueError("Expected CUDA device.")
        return device.index if device.index is not None else torch.cuda.current_device()
    return int(device)


@cython.locals(entry=cython.int)
def _get_cache_slot(int device_index) -> int:
    cdef int i
    global _mem_cache_size
    for i in range(_mem_cache_size):
        if _mem_cache_devices[i] == device_index:
            return i
    if _mem_cache_size < _MAX_MEM_CACHE:
        entry = _mem_cache_size
        _mem_cache_size += 1
    else:
        entry = _mem_cache_size - 1
    _mem_cache_devices[entry] = device_index
    _mem_cache_timestamp[entry] = 0.0
    _mem_cache_value[entry] = 0.0
    return entry


def _cached_available_bytes(device: torch.device, optim: Optional[MemoryOptimizationConfig]) -> float:
    if optim is None or not optim.cache_memory_stats:
        memory_stats = torch.cuda.memory_stats(device)
        bytes_active = memory_stats['active_bytes.all.current']
        bytes_reserved = memory_stats['reserved_bytes.all.current']
        bytes_free_cuda, _ = torch.cuda.mem_get_info(device)
        bytes_inactive_reserved = bytes_reserved - bytes_active
        return bytes_free_cuda + bytes_inactive_reserved

    ttl = float(max(0.0, optim.stats_cache_ttl))
    idx = _device_index(device)
    entry = _get_cache_slot(idx)
    now = time.perf_counter()
    elapsed = now - _mem_cache_timestamp[entry]
    if elapsed <= ttl:
        return _mem_cache_value[entry]

    memory_stats = torch.cuda.memory_stats(device)
    bytes_active = memory_stats['active_bytes.all.current']
    bytes_reserved = memory_stats['reserved_bytes.all.current']
    bytes_free_cuda, _ = torch.cuda.mem_get_info(device)
    bytes_inactive_reserved = bytes_reserved - bytes_active
    total = bytes_free_cuda + bytes_inactive_reserved
    _mem_cache_timestamp[entry] = now
    _mem_cache_value[entry] = total
    return total


class DynamicSwapInstaller:
    @staticmethod
    def _install_module(module: torch.nn.Module, **kwargs: Any) -> None:
        original_class = module.__class__
        module.__dict__['forge_backup_original_class'] = original_class

        def hacked_get_attr(self, name: str):
            if '_parameters' in self.__dict__:
                params = self.__dict__['_parameters']
                value = params.get(name)
                if value is not None:
                    if isinstance(value, torch.nn.Parameter):
                        return torch.nn.Parameter(value.to(**kwargs), requires_grad=value.requires_grad)
                    return value.to(**kwargs)
            if '_buffers' in self.__dict__:
                buf = self.__dict__['_buffers'].get(name)
                if buf is not None:
                    return buf.to(**kwargs)
            return super(original_class, self).__getattr__(name)

        module.__class__ = type('DynamicSwap_' + original_class.__name__, (original_class,), {
            '__getattr__': hacked_get_attr,
        })

    @staticmethod
    def _uninstall_module(module: torch.nn.Module) -> None:
        original = module.__dict__.pop('forge_backup_original_class', None)
        if original is not None:
            module.__class__ = original

    @staticmethod
    def install_model(model: torch.nn.Module, **kwargs: Any) -> None:
        for sub in model.modules():
            DynamicSwapInstaller._install_module(sub, **kwargs)

    @staticmethod
    def uninstall_model(model: torch.nn.Module) -> None:
        for sub in model.modules():
            DynamicSwapInstaller._uninstall_module(sub)


def fake_diffusers_current_device(model: torch.nn.Module, target_device: torch.device) -> None:
    if hasattr(model, 'scale_shift_table'):
        model.scale_shift_table.data = model.scale_shift_table.data.to(target_device)
        return

    for _, sub in model.named_modules():
        if hasattr(sub, 'weight'):
            sub.to(target_device)
            return


def get_cuda_free_memory_gb(
    device: Optional[torch.device] = None,
    optim_config: Optional[MemoryOptimizationConfig] = None,
) -> float:
    target = device or gpu
    available_bytes = _cached_available_bytes(target, optim_config)
    return available_bytes / (1024 ** 3)


def move_model_to_device_with_memory_preservation(
    model: torch.nn.Module,
    target_device: torch.device,
    preserved_memory_gb: float = 0,
    aggressive: bool = False,
    optim_config: Optional[MemoryOptimizationConfig] = None,
) -> None:
    modules_list = list(model.modules())
    for idx, module in enumerate(modules_list):
        free_mem = get_cuda_free_memory_gb(target_device, optim_config)
        if free_mem <= preserved_memory_gb:
            torch.cuda.empty_cache()
            if not aggressive:
                return
            torch.cuda.synchronize()
            continue
        if hasattr(module, 'weight'):
            module.to(device=target_device)
            if aggressive and idx % 10 == 0:
                torch.cuda.empty_cache()
    model.to(device=target_device)
    torch.cuda.empty_cache()


def offload_model_from_device_for_memory_preservation(
    model: torch.nn.Module,
    target_device: torch.device,
    preserved_memory_gb: float = 0,
    aggressive: bool = False,
    optim_config: Optional[MemoryOptimizationConfig] = None,
) -> None:
    modules_list = list(model.modules())
    for idx, module in enumerate(modules_list):
        free_mem = get_cuda_free_memory_gb(target_device, optim_config)
        if free_mem >= preserved_memory_gb:
            if not aggressive:
                torch.cuda.empty_cache()
                return
        if hasattr(module, 'weight'):
            module.to(device=cpu)
            if aggressive and idx % 10 == 0:
                torch.cuda.empty_cache()
    model.to(device=cpu)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def unload_complete_models(*models: torch.nn.Module) -> None:
    for module in list(gpu_complete_modules) + list(models):
        module.to(device=cpu)
    gpu_complete_modules.clear()
    torch.cuda.empty_cache()


def load_model_as_complete(model: torch.nn.Module, target_device: torch.device, unload: bool = True) -> None:
    if unload:
        unload_complete_models()
    model.to(device=target_device)
    gpu_complete_modules.append(model)


@cython.locals(chunk_bytes=cython.Py_ssize_t, elem_bytes=cython.Py_ssize_t, elems=cython.Py_ssize_t)
def _calc_chunk_elems(chunk_bytes: int, elem_bytes: int) -> int:
    if elem_bytes <= 0:
        return 1
    elems = chunk_bytes // elem_bytes
    if elems < 1:
        return 1
    return elems


@cython.locals(current=cython.Py_ssize_t, limit=cython.Py_ssize_t, stride=cython.Py_ssize_t, nxt=cython.Py_ssize_t)
def _next_offset(current: int, limit: int, stride: int) -> int:
    nxt = current + stride
    if nxt > limit:
        return limit
    return nxt


@cython.locals(
    chunk_elems=cython.Py_ssize_t,
    non_blocking=cython.bint,
    start=cython.Py_ssize_t,
    end=cython.Py_ssize_t,
    total=cython.Py_ssize_t,
)
def _copy_chunked_tensor(
    flat_src: torch.Tensor,
    flat_dst: torch.Tensor,
    chunk_elems: int,
    non_blocking: bool,
) -> None:
    start = 0
    total = flat_src.numel()
    while start < total:
        end = _next_offset(start, total, chunk_elems)
        dst_view = flat_dst.narrow(0, start, end - start)
        src_view = flat_src.narrow(0, start, end - start)
        dst_view.copy_(src_view, non_blocking=non_blocking)
        start = end


def load_model_chunked(
    model: torch.nn.Module,
    target_device: torch.device,
    max_chunk_size_mb: int = 256,
    optim_config: Optional[MemoryOptimizationConfig] = None,
) -> None:
    optim = optim_config or MemoryOptimizationConfig()
    unload_complete_models()

    chunk_bytes = max(1, int(max_chunk_size_mb) * 1024 * 1024)
    modules = list(model.modules())

    for idx, module in enumerate(modules):
        if idx % 50 == 0:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        tensors: Iterable[torch.Tensor] = list(module.parameters(recurse=False)) + list(module.buffers(recurse=False))
        for tensor in tensors:
            if tensor is None or tensor.device == target_device:
                continue
            contiguous = tensor.contiguous()
            if optim.use_pinned_memory and not contiguous.is_cuda:
                contiguous = contiguous.pin_memory()

            elem_bytes = contiguous.element_size()
            total_elems = contiguous.numel()
            chunk_elems = _calc_chunk_elems(chunk_bytes, elem_bytes)

            dst = torch.empty_like(contiguous, device=target_device)
            flat_src = contiguous.view(-1)
            flat_dst = dst.view(-1)

            if optim.enable_async_copy():
                copy_stream = torch.cuda.Stream(device=target_device)
                with torch.cuda.stream(copy_stream):
                    _copy_chunked_tensor(flat_src, flat_dst, chunk_elems, True)
                torch.cuda.current_stream(device=target_device).wait_stream(copy_stream)
            else:
                _copy_chunked_tensor(flat_src, flat_dst, chunk_elems, False)

            tensor.data = dst

        if idx % 5 == 0:
            torch.cuda.empty_cache()

    gpu_complete_modules.append(model)
    torch.cuda.empty_cache()


def force_free_vram(target_gb: float = 2.0, optim_config: Optional[MemoryOptimizationConfig] = None) -> float:
    unload_complete_models()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    free_mem = get_cuda_free_memory_gb(gpu, optim_config=optim_config)
    if free_mem < target_gb:
        print(f'Warning: only {free_mem:.2f} GB free, requested {target_gb} GB')
    return free_mem
