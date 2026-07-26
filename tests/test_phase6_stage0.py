from __future__ import annotations

import json

import torch

from kam.memory import SparseMemoryConfig, SparseSeparableMemory
from kam.memory.routers import ChunkedExactTopKRouter, ExactTopKRouter
from kam.optimization import AlternatingSchedule, ridge_solve, streaming_rls
from kam.phase6.manifest import build_stage0_rows, load_config
from kam.phase6.diagnostics import measure_forward
from kam.phase6.resource import forecast_transformer_memory
from kam.transformer import ModernDecoder, TransformerConfig


def test_phase6_chunked_router_matches_exact() -> None:
    torch.manual_seed(1)
    query = torch.randn(9, 8)
    keys = torch.randn(37, 8)
    exact = ExactTopKRouter(5)(query, keys)
    chunked = ChunkedExactTopKRouter(5, chunk_size=6)(query, keys)
    assert torch.equal(exact.indices.sort(-1).values, chunked.indices.sort(-1).values)
    assert torch.allclose(exact.weights, chunked.weights, atol=1e-6)


def test_phase6_memory_backward_and_zero_gate() -> None:
    memory = SparseSeparableMemory(SparseMemoryConfig(d_model=8, num_supports=12, top_k=3, expert_mode="low_rank", expert_rank=2), seed=3)
    query = torch.randn(2, 5, 8, requires_grad=True)
    loss = memory(query).square().mean()
    # The output is zero at the zero gate, so use a target to exercise the gate.
    loss = (memory(query) - 0.25).square().mean()
    loss.backward()
    assert memory.gate.logit.grad is not None
    assert torch.isfinite(memory.gate.logit.grad)
    assert torch.equal(memory(query), torch.zeros_like(query))


def test_phase6_zero_gate_preserves_transformer_logits() -> None:
    config = TransformerConfig(d_model=8, n_heads=2, n_layers=1, vocab_size=13, max_seq_len=8)
    torch.manual_seed(5)
    baseline = ModernDecoder(config)
    memory = SparseSeparableMemory(SparseMemoryConfig(d_model=8, num_supports=10, top_k=2), seed=5)
    with_memory = ModernDecoder(config, memory_layers=[memory])
    with_memory.load_state_dict(baseline.state_dict(), strict=False)
    tokens = torch.randint(13, (2, 8))
    assert torch.allclose(baseline(tokens), with_memory(tokens), atol=1e-6)


def test_phase6_ridge_and_streaming_rls_agree() -> None:
    torch.manual_seed(7)
    x = torch.randn(30, 4, dtype=torch.float64)
    y = torch.randn(30, dtype=torch.float64)
    direct = ridge_solve(x, y, regularization=1e-2)
    online = streaming_rls(x, y, regularization=1e-2)
    assert torch.allclose(direct.solution, online.solution, atol=1e-6)


def test_phase6_manifest_is_deterministic_and_stage0_sized() -> None:
    config = load_config("configs/phase6/stage0_validity.yaml")
    first = build_stage0_rows(config)
    second = build_stage0_rows(config)
    assert len(first) == 128
    assert first == second
    assert len({row["row_id"] for row in first}) == len(first)


def test_phase6_resource_forecast_and_schedule() -> None:
    forecast = forecast_transformer_memory(
        d_model=16,
        n_layers=2,
        d_ff=64,
        num_supports=32,
        top_k=4,
        expert_mode="low_rank",
    )
    assert forecast.memory_parameters > forecast.active_parameters_per_token
    assert AlternatingSchedule.from_label("alternating_8:1").phase(8) == "geometry"


def test_phase6_forward_measurement_is_finite() -> None:
    memory = SparseSeparableMemory(SparseMemoryConfig(d_model=8, num_supports=10, top_k=2), seed=9)
    metrics = measure_forward(memory, batch_size=1, sequence_length=4, d_model=8, repeats=2, warmup=1)
    assert metrics["forward_median_ms"] > 0
    assert metrics["throughput_tokens_per_sec"] > 0
