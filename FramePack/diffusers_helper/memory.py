# By lllyasviel


import torch


cpu = torch.device('cpu')
gpu = torch.device(f'cuda:{torch.cuda.current_device()}')
gpu_complete_modules = []


class DynamicSwapInstaller:
    @staticmethod
    def _install_module(module: torch.nn.Module, **kwargs):
        original_class = module.__class__
        module.__dict__['forge_backup_original_class'] = original_class

        def hacked_get_attr(self, name: str):
            if '_parameters' in self.__dict__:
                _parameters = self.__dict__['_parameters']
                if name in _parameters:
                    p = _parameters[name]
                    if p is None:
                        return None
                    if p.__class__ == torch.nn.Parameter:
                        return torch.nn.Parameter(p.to(**kwargs), requires_grad=p.requires_grad)
                    else:
                        return p.to(**kwargs)
            if '_buffers' in self.__dict__:
                _buffers = self.__dict__['_buffers']
                if name in _buffers:
                    return _buffers[name].to(**kwargs)
            return super(original_class, self).__getattr__(name)

        module.__class__ = type('DynamicSwap_' + original_class.__name__, (original_class,), {
            '__getattr__': hacked_get_attr,
        })

        return

    @staticmethod
    def _uninstall_module(module: torch.nn.Module):
        if 'forge_backup_original_class' in module.__dict__:
            module.__class__ = module.__dict__.pop('forge_backup_original_class')
        return

    @staticmethod
    def install_model(model: torch.nn.Module, **kwargs):
        for m in model.modules():
            DynamicSwapInstaller._install_module(m, **kwargs)
        return

    @staticmethod
    def uninstall_model(model: torch.nn.Module):
        for m in model.modules():
            DynamicSwapInstaller._uninstall_module(m)
        return


def fake_diffusers_current_device(model: torch.nn.Module, target_device: torch.device):
    if hasattr(model, 'scale_shift_table'):
        model.scale_shift_table.data = model.scale_shift_table.data.to(target_device)
        return

    for k, p in model.named_modules():
        if hasattr(p, 'weight'):
            p.to(target_device)
            return


def get_cuda_free_memory_gb(device=None):
    if device is None:
        device = gpu

    memory_stats = torch.cuda.memory_stats(device)
    bytes_active = memory_stats['active_bytes.all.current']
    bytes_reserved = memory_stats['reserved_bytes.all.current']
    bytes_free_cuda, _ = torch.cuda.mem_get_info(device)
    bytes_inactive_reserved = bytes_reserved - bytes_active
    bytes_total_available = bytes_free_cuda + bytes_inactive_reserved
    return bytes_total_available / (1024 ** 3)


def move_model_to_device_with_memory_preservation(model, target_device, preserved_memory_gb=0, aggressive=False):
    print(f'Moving {model.__class__.__name__} to {target_device} with preserved memory: {preserved_memory_gb} GB')

    modules_list = list(model.modules())

    for i, m in enumerate(modules_list):
        free_mem = get_cuda_free_memory_gb(target_device)

        if free_mem <= preserved_memory_gb:
            if aggressive:
                # Aggressive mode: clear cache and try to continue
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                free_mem = get_cuda_free_memory_gb(target_device)

                if free_mem <= preserved_memory_gb * 0.8:
                    print(f'Stopped at module {i}/{len(modules_list)} due to memory limit')
                    return
            else:
                torch.cuda.empty_cache()
                return

        if hasattr(m, 'weight'):
            m.to(device=target_device)

            # Clear cache every 10 modules
            if aggressive and i % 10 == 0:
                torch.cuda.empty_cache()

    model.to(device=target_device)
    torch.cuda.empty_cache()
    return


def offload_model_from_device_for_memory_preservation(model, target_device, preserved_memory_gb=0, aggressive=False):
    print(f'Offloading {model.__class__.__name__} from {target_device} to preserve memory: {preserved_memory_gb} GB')

    modules_list = list(model.modules())

    for i, m in enumerate(modules_list):
        free_mem = get_cuda_free_memory_gb(target_device)

        if free_mem >= preserved_memory_gb:
            if not aggressive:
                torch.cuda.empty_cache()
                return

        if hasattr(m, 'weight'):
            m.to(device=cpu)

            # Clear cache every 10 modules
            if aggressive and i % 10 == 0:
                torch.cuda.empty_cache()

    model.to(device=cpu)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return


def unload_complete_models(*args):
    for m in gpu_complete_modules + list(args):
        m.to(device=cpu)
        print(f'Unloaded {m.__class__.__name__} as complete.')

    gpu_complete_modules.clear()
    torch.cuda.empty_cache()
    return


def load_model_as_complete(model, target_device, unload=True):
    if unload:
        unload_complete_models()

    model.to(device=target_device)
    print(f'Loaded {model.__class__.__name__} to {target_device} as complete.')

    gpu_complete_modules.append(model)
    return


def load_model_chunked(model, target_device, max_chunk_size_mb=512):
    """
    Load model to device in smaller chunks to bypass memory limits.
    Useful for extremely large models (110GB+) on limited VRAM (16GB).
    """
    print(f'Loading {model.__class__.__name__} to {target_device} in chunks (max {max_chunk_size_mb}MB per chunk)')

    unload_complete_models()

    modules = list(model.modules())
    total_modules = len(modules)

    for i, module in enumerate(modules):
        if i % 50 == 0:
            print(f'Loading module {i}/{total_modules}...')
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Skip modules without parameters
        if not any(p.numel() > 0 for p in module.parameters(recurse=False)):
            continue

        # Move module to target device
        try:
            module.to(device=target_device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f'OOM at module {i}, clearing cache and retrying...')
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # Try again after clearing
                try:
                    module.to(device=target_device)
                except RuntimeError:
                    print(f'Failed to load module {i}, keeping on CPU')
                    module.to(device=cpu)
            else:
                raise

        # Clear cache after every module
        if i % 5 == 0:
            torch.cuda.empty_cache()

    print(f'Finished loading {model.__class__.__name__}')
    gpu_complete_modules.append(model)
    torch.cuda.empty_cache()
    return


def force_free_vram(target_gb=2.0):
    """
    Aggressively free VRAM until target_gb is available.
    """
    print(f'Force freeing VRAM to reach {target_gb} GB...')

    # First, unload all complete models
    unload_complete_models()

    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    free_mem = get_cuda_free_memory_gb(gpu)
    print(f'After clearing: {free_mem:.2f} GB free')

    if free_mem < target_gb:
        print(f'Warning: Only {free_mem:.2f} GB available, requested {target_gb} GB')

    return free_mem
