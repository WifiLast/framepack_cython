import dataclasses
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class KVCachingConfig:
    enabled: bool = True
    max_length: int = 2048
    reset_on_mismatch: bool = True
    verbose: bool = False


class KVCachingManager:
    def __init__(self, config: KVCachingConfig):
        self.config = config
        self.caches: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.position_offset = 0

    def reset(self):
        self.caches.clear()
        self.position_offset = 0

    def get(self, key: str):
        return self.caches.get(key)

    def update(self, key: str, k: torch.Tensor, v: torch.Tensor):
        if not self.config.enabled:
            return
        max_len = self.config.max_length
        k = k.detach()
        v = v.detach()
        if key in self.caches:
            prev_k, prev_v = self.caches[key]
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
        if k.shape[2] > max_len:
            k = k[:, :, -max_len:, :]
            v = v[:, :, -max_len:, :]
        self.caches[key] = (k, v)

    def append(self, key: str, new_k: torch.Tensor, new_v: torch.Tensor):
        self.update(key, new_k, new_v)

    def state_dict(self):
        serialized = {}
        for key, (k, v) in self.caches.items():
            serialized[key] = {
                "k": k.detach().cpu(),
                "v": v.detach().cpu(),
            }
        return {
            "config": dataclasses.asdict(self.config) if self.config else None,
            "position_offset": self.position_offset,
            "caches": serialized,
        }

    def load_state_dict(self, state_dict):
        if not isinstance(state_dict, dict):
            return
        config_data = state_dict.get("config")
        if config_data:
            self.config = KVCachingConfig(**config_data)
        self.position_offset = int(state_dict.get("position_offset", 0))
        caches = state_dict.get("caches", {})
        self.caches = {}
        if isinstance(caches, dict):
            for key, tensors in caches.items():
                if not isinstance(tensors, dict):
                    continue
                k = tensors.get("k")
                v = tensors.get("v")
                if k is None or v is None:
                    continue
                self.caches[key] = (k, v)

