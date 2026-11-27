"""Parameter-matched residual KAM and causal dot-attention adapters."""
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from kam.memory.experts import VectorExperts
from kam.memory.routers import ExactTopKRouter

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
LAYERS = (7, 14, 21, 27)
WIDTH = 64
HEADS = 4
SUPPORTS = WIDTH * 4
ARMS = ("kam", "dot", "pretrained")
SEED = 731201


class ResidualAdapter(nn.Module):
    def __init__(self, hidden, arm, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.down = nn.Linear(hidden, WIDTH, bias=False)
            self.up = nn.Linear(WIDTH, hidden, bias=False)
            nn.init.zeros_(self.up.weight)
            if arm == "dot":
                self.qkv = nn.Linear(WIDTH, 3 * WIDTH, bias=False)
                self.out = nn.Linear(WIDTH, WIDTH, bias=False)
            else:
                self.register_buffer("keys", F.normalize(torch.randn(SUPPORTS, WIDTH), dim=-1))
                self.router = ExactTopKRouter(top_k=4, metric="dot")
                self.experts = VectorExperts(SUPPORTS, WIDTH)
        self.arm = arm
        self.mask = None
        self.cache_enabled = False
        self.cache = None

    def forward(self, x):
        z = self.down(F.rms_norm(x, (x.shape[-1],)))
        if self.arm == "kam":
            flat = z.reshape(-1, WIDTH)
            value = self.experts(flat, self.router(flat, self.keys)).reshape_as(z)
        else:
            q, k, v = (r.unflatten(-1, (HEADS, WIDTH // HEADS)).transpose(1, 2)
                       for r in self.qkv(z).chunk(3, dim=-1))
            old = self.cache[0].size(-2) if self.cache is not None else 0
            if old:
                k = torch.cat((self.cache[0], k), dim=-2)
                v = torch.cat((self.cache[1], v), dim=-2)
            if self.cache_enabled:
                self.cache = (k, v)
            mask = None
            if self.mask is not None:
                mask = self.mask[:, None, None, :k.size(-2)]
                if q.size(-2) > 1:
                    keys = torch.arange(k.size(-2), device=q.device)
                    queries = torch.arange(old, old + q.size(-2), device=q.device)
                    mask = mask & (keys[None, :] <= queries[:, None])
            value = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                   is_causal=mask is None and not old)
            value = self.out(value.transpose(1, 2).reshape_as(z))
        return x + self.up(value).to(x.dtype)


class AdapterStack:
    def __init__(self, model, arm):
        self.modules = nn.ModuleList()
        self.handles = []
        if arm == "pretrained":
            return
        for index in LAYERS:
            adapter = ResidualAdapter(model.config.hidden_size, arm, SEED + index)
            self.modules.append(adapter)
            self.handles.append(model.model.layers[index].register_forward_hook(self.hook(adapter)))
        self.modules.cuda()
        expected = len(LAYERS) * (2 * model.config.hidden_size * WIDTH + 4 * WIDTH**2)
        assert sum(p.numel() for p in self.modules.parameters()) == expected

    @staticmethod
    def hook(adapter):
        def apply(_layer, _inputs, output):
            if isinstance(output, tuple):
                return (adapter(output[0]), *output[1:])
            return adapter(output)
        return apply

    def context(self, mask, use_cache):
        for adapter in self.modules:
            adapter.mask = mask
            adapter.cache_enabled = use_cache
            if not use_cache:
                adapter.cache = None

    def reset(self):
        self.context(None, False)

    def close(self):
        for handle in self.handles:
            handle.remove()


def load_model(root: Path):
    tokenizer = AutoTokenizer.from_pretrained(root / "model", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        root / "model", dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa"
    ).cuda().eval()
    model.requires_grad_(False)
    return model, tokenizer


@contextmanager
def mixed_precision():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        yield
