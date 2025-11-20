import math
import dataclasses
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:  # pragma: no cover
    faiss = None
    _HAS_FAISS = False


@dataclass
class SimilarityCacheConfig:
    enabled: bool = True
    threshold: float = 0.9
    max_skip: int = 1
    max_entries: int = 16
    use_faiss: bool = False
    max_age: int = 32
    verbose: bool = False


class LearnableProjector(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cached_hidden: torch.Tensor, similarity: torch.Tensor) -> torch.Tensor:
        projected = self.linear(cached_hidden)
        similarity = similarity.clamp(0.0, 1.0)
        while similarity.dim() < projected.dim():
            similarity = similarity.unsqueeze(-1)
        return cached_hidden + similarity * (projected - cached_hidden)


class BlockCacheEntry:
    def __init__(self, key: torch.Tensor, hidden: torch.Tensor, encoder: Optional[torch.Tensor], step: int):
        self.key = F.normalize(key.float(), dim=-1)
        self.hidden = hidden.detach().cpu()
        self.encoder = encoder.detach().cpu() if encoder is not None else None
        self.step = step


class BlockSimilarityCache:
    def __init__(self, key_dim: int, max_entries: int = 16, use_faiss: bool = False, max_age: int = 32):
        self.key_dim = key_dim
        self.max_entries = max_entries
        self.max_age = max_age
        self.entries: list[BlockCacheEntry] = []
        self.use_faiss = use_faiss and _HAS_FAISS
        self._faiss_index = None
        if self.use_faiss:
            self._faiss_index = faiss.IndexFlatIP(key_dim)

    def _rebuild_index(self):
        if not self.use_faiss:
            return
        self._faiss_index.reset()
        if not self.entries:
            return
        keys = torch.stack([entry.key for entry in self.entries]).numpy()
        self._faiss_index.add(keys)

    def lookup(self, key: torch.Tensor) -> Optional[Tuple[BlockCacheEntry, float]]:
        if not self.entries:
            return None
        key = F.normalize(key.float(), dim=-1)
        if self.use_faiss and self._faiss_index is not None:
            distances, indices = self._faiss_index.search(key.cpu().numpy()[None, :], 1)
            idx = int(indices[0][0])
            if idx < 0 or idx >= len(self.entries):
                return None
            sim = float(distances[0][0])
            return self.entries[idx], sim
        stacked = torch.stack([entry.key for entry in self.entries])
        sims = torch.mv(stacked, key.cpu())
        max_sim, idx = torch.max(sims, dim=0)
        return self.entries[int(idx)], float(max_sim.item())

    def update(self, key: torch.Tensor, hidden: torch.Tensor, encoder: Optional[torch.Tensor], step: int):
        entry = BlockCacheEntry(key, hidden, encoder, step)
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)
        if self.use_faiss:
            self._rebuild_index()

    def prune(self, current_step: int):
        if self.max_age <= 0 or not self.entries:
            return
        keep = [entry for entry in self.entries if current_step - entry.step <= self.max_age]
        if len(keep) != len(self.entries):
            self.entries = keep
            if self.use_faiss:
                self._rebuild_index()

    def clear(self):
        self.entries.clear()
        if self.use_faiss and self._faiss_index is not None:
            self._faiss_index.reset()

    def state_dict(self):
        serialized_entries = []
        for entry in self.entries:
            serialized_entries.append(
                {
                    "key": entry.key.cpu(),
                    "hidden": entry.hidden.cpu(),
                    "encoder": entry.encoder.cpu() if entry.encoder is not None else None,
                    "step": entry.step,
                }
            )
        return {
            "max_entries": self.max_entries,
            "max_age": self.max_age,
            "entries": serialized_entries,
        }

    def load_state_dict(self, state_dict):
        self.clear()
        if not isinstance(state_dict, dict):
            return
        entries = state_dict.get("entries", [])
        for entry in entries:
            key = entry.get("key")
            hidden = entry.get("hidden")
            step = entry.get("step", 0)
            if key is None or hidden is None:
                continue
            encoder = entry.get("encoder")
            new_entry = BlockCacheEntry(key, hidden, encoder, step)
            self.entries.append(new_entry)
        if self.use_faiss:
            self._rebuild_index()


class SimilarityCacheManager:
    def __init__(
        self,
        num_dual_blocks: int,
        num_single_blocks: int,
        hidden_dim: int,
        config: SimilarityCacheConfig,
    ):
        self.config = config
        self.dual_caches = [
            BlockSimilarityCache(hidden_dim, config.max_entries, config.use_faiss, config.max_age)
            for _ in range(num_dual_blocks)
        ]
        self.single_caches = [
            BlockSimilarityCache(hidden_dim, config.max_entries, config.use_faiss, config.max_age)
            for _ in range(num_single_blocks)
        ]
        self.dual_steps = [0] * num_dual_blocks
        self.single_steps = [0] * num_single_blocks

    def _step_tracker(self, block_type: str):
        return self.dual_steps if block_type == "dual" else self.single_steps

    def step(self, block_type: str, block_id: int) -> int:
        steps = self._step_tracker(block_type)
        steps[block_id] += 1
        return steps[block_id]

    def current_step(self, block_type: str, block_id: int) -> int:
        return self._step_tracker(block_type)[block_id]

    def get(self, block_type: str, block_id: int) -> BlockSimilarityCache:
        if block_type == "dual":
            return self.dual_caches[block_id]
        return self.single_caches[block_id]

    def prune(self, block_type: Optional[str] = None, block_id: Optional[int] = None):
        if block_type is not None and block_id is not None:
            cache = self.get(block_type, block_id)
            cache.prune(self.current_step(block_type, block_id))
            return

        for idx, cache in enumerate(self.dual_caches):
            cache.prune(self.dual_steps[idx])
        for idx, cache in enumerate(self.single_caches):
            cache.prune(self.single_steps[idx])

    def clear(self):
        for cache in self.dual_caches + self.single_caches:
            cache.clear()
        self.dual_steps = [0] * len(self.dual_caches)
        self.single_steps = [0] * len(self.single_caches)

    def state_dict(self):
        max_step = 0
        if self.dual_steps:
            max_step = max(max_step, max(self.dual_steps))
        if self.single_steps:
            max_step = max(max_step, max(self.single_steps))
        return {
            "config": dataclasses.asdict(self.config) if self.config else None,
            "global_step": max_step,
            "dual_steps": list(self.dual_steps),
            "single_steps": list(self.single_steps),
            "dual_caches": [cache.state_dict() for cache in self.dual_caches],
            "single_caches": [cache.state_dict() for cache in self.single_caches],
        }

    def load_state_dict(self, state_dict):
        if not isinstance(state_dict, dict):
            return
        config_data = state_dict.get("config")
        if config_data:
            self.config = SimilarityCacheConfig(**config_data)
        legacy_step = int(state_dict.get("global_step", 0))
        dual_steps = state_dict.get("dual_steps")
        single_steps = state_dict.get("single_steps")
        if not isinstance(dual_steps, list):
            dual_steps = [legacy_step] * len(self.dual_caches)
        if not isinstance(single_steps, list):
            single_steps = [legacy_step] * len(self.single_caches)
        self.dual_steps = [int(dual_steps[i]) if i < len(dual_steps) else legacy_step for i in range(len(self.dual_caches))]
        self.single_steps = [
            int(single_steps[i]) if i < len(single_steps) else legacy_step
            for i in range(len(self.single_caches))
        ]
        dual_states = state_dict.get("dual_caches", [])
        single_states = state_dict.get("single_caches", [])
        for cache, cache_state in zip(self.dual_caches, dual_states):
            cache.load_state_dict(cache_state)
        for cache, cache_state in zip(self.single_caches, single_states):
            cache.load_state_dict(cache_state)
