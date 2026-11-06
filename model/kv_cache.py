import torch
import torch.nn as nn
from typing import Tuple, Optional

class KVCache:
    def __init__(self):
        self.cache_k: Optional[torch.Tensor] = None  # (batch, seq_len, model_dim)
        self.cache_v: Optional[torch.Tensor] = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Append new_k and new_v to the cache along the sequence dimension (dim=1).
        # On the first call, initialize the cache with the given tensors.
        # Return the full (cached) K and V tensors.
        # First call: initialize cache
        if self.cache_k is None:
            self.cache_k = new_k
            self.cache_v = new_v

        else:
            # Append new K and V along sequence dimension
            self.cache_k = torch.cat(
                (self.cache_k, new_k), dim=1
            )

            self.cache_v = torch.cat(
                (self.cache_v, new_v), dim=1
            )

        return self.cache_k, self.cache_v

    def clear(self):
        self.cache_k = None
        self.cache_v = None

class CachedAttention(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> Tuple[torch.Tensor, KVCache]:
        # 1. Project x into Q, K, V using the linear layers
        # 2. If kv_cache is None, create a new KVCache
        # 3. Update the cache with the new K and V
        # 4. Compute scaled dot-product attention using Q and the full cached K, V
        # 5. Apply a causal mask offset by the number of previously cached tokens
        # 6. Return (rounded output, kv_cache)
        # 1. Project input into Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Create cache if not provided
        if kv_cache is None:
            kv_cache = KVCache()

        # 3. Update cache
        previous_length = (
            0 if kv_cache.cache_k is None
            else kv_cache.cache_k.size(1)
        )

        cached_k, cached_v = kv_cache.update(k, v)

        # 4. Scaled dot-product attention
        scores = q @ cached_k.transpose(1, 2)

        scores = scores / (q.size(-1) ** 0.5)

        # 5. Causal mask with offset
        query_length = q.size(1)
        key_length = cached_k.size(1)

        query_positions = (
            torch.arange(query_length, device=x.device)
            + previous_length
        )

        key_positions = torch.arange(key_length, device=x.device)

        mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)

        scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))

        # 6. Softmax and weighted sum
        attention_weights = torch.softmax(scores, dim=-1)

        output = attention_weights @ cached_v

        # Round output to 4 decimal places
        output = torch.round(output * 10000) / 10000

        return output, kv_cache

