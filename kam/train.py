from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .data import CharacterDataset, CopyLanguageDataset, RegimeGrammarDataset, load_text, make_mackey_splits
from .factory import KAM_MODELS, make_model
from .model import KAMSequenceModel, ModelDiagnostics
from .utils import atomic_torch_save, choose_device, save_json, set_seed


@dataclass
class DataBundle:
    train_loader: DataLoader
    validation_loader: DataLoader
    task_type: str
    model_fields: dict[str, Any]
    metadata: dict[str, Any]


def masked_language_loss(logits: Tensor, targets: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    flat_loss = nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    ).reshape_as(targets)
    mask = mask.to(dtype=flat_loss.dtype)
    denominator = mask.sum().clamp_min(1.0)
    loss = (flat_loss * mask).sum() / denominator
    predictions = logits.argmax(dim=-1)
    accuracy = (((predictions == targets).to(mask.dtype) * mask).sum() / denominator)
    return loss, accuracy


def _make_language_loaders(dataset: Any, validation: Any, batch_size: int) -> tuple[DataLoader, DataLoader]:
    return (
        DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def build_data(args: argparse.Namespace) -> DataBundle:
    if args.task == "mackey-glass":
        splits = make_mackey_splits(
            total_length=args.series_length,
            window=args.seq_len,
            tau=args.tau,
            beta=args.beta,
            seed=args.seed,
        )
        train_loader = DataLoader(splits.train, batch_size=args.batch_size, shuffle=True, num_workers=0)
        validation_loader = DataLoader(
            splits.validation, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        return DataBundle(
            train_loader=train_loader,
            validation_loader=validation_loader,
            task_type="regression",
            model_fields={"input_dim": 2, "output_dim": 1, "max_seq_len": args.seq_len},
            metadata={"mean": splits.mean, "std": splits.std, "tau": args.tau, "beta": args.beta},
        )

    if args.task == "copy":
        train = CopyLanguageDataset(
            size=args.train_size,
            payload_length=args.copy_length,
            alphabet_size=args.alphabet_size,
            seed=args.seed,
        )
        validation = CopyLanguageDataset(
            size=args.val_size,
            payload_length=args.copy_length,
            alphabet_size=args.alphabet_size,
            seed=args.seed + 1_000_000,
        )
        train_loader, validation_loader = _make_language_loaders(train, validation, args.batch_size)
        return DataBundle(
            train_loader,
            validation_loader,
            "language",
            {"vocab_size": train.vocab_size, "max_seq_len": train.sequence_length},
            {"vocab_size": train.vocab_size, "copy_length": args.copy_length, "alphabet_size": args.alphabet_size},
        )

    if args.task == "regime":
        train = RegimeGrammarDataset(
            size=args.train_size,
            sequence_length=args.seq_len,
            alphabet_size=args.alphabet_size,
            num_regimes=args.num_regimes,
            seed=args.seed,
            switch_halfway=False,
        )
        validation = RegimeGrammarDataset(
            size=args.val_size,
            sequence_length=args.seq_len,
            alphabet_size=args.alphabet_size,
            num_regimes=args.num_regimes,
            seed=args.seed + 1_000_000,
            switch_halfway=args.regime_switch_validation,
        )
        train_loader, validation_loader = _make_language_loaders(train, validation, args.batch_size)
        return DataBundle(
            train_loader,
            validation_loader,
            "language",
            {"vocab_size": train.vocab_size, "max_seq_len": args.seq_len + 1},
            {
                "vocab_size": train.vocab_size,
                "num_regimes": args.num_regimes,
                "alphabet_size": args.alphabet_size,
                "sequence_length": args.seq_len,
                "switch_validation": args.regime_switch_validation,
            },
        )

    if args.task == "char":
        text = load_text(args.text_path)
        split = int(0.90 * len(text))
        train = CharacterDataset(text[:split], sequence_length=args.seq_len, stride=args.char_stride)
        validation_text = text[max(0, split - args.seq_len - 1) :]
        validation = CharacterDataset(
            validation_text,
            sequence_length=args.seq_len,
            stride=args.char_stride,
            vocabulary=train.characters,
        )
        train_loader, validation_loader = _make_language_loaders(train, validation, args.batch_size)
        return DataBundle(
            train_loader,
            validation_loader,
            "language",
            {"vocab_size": train.vocab_size, "max_seq_len": args.seq_len},
            {"vocab_size": train.vocab_size, "text_path": str(args.text_path)},
        )

    raise ValueError(f"Unsupported task: {args.task}")


def build_model_spec(args: argparse.Namespace, data: DataBundle) -> dict[str, Any]:
    if args.model == "mlp" and data.task_type != "regression":
        raise ValueError("The MLP baseline is available only for Mackey-Glass regression.")
    spec = {
        "model_name": args.model,
        "task_type": data.task_type,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "num_supports": args.num_supports,
        "context_window": args.context_window,
        "dropout": args.dropout,
        "expose_memory_weights": not args.hide_memory_weights,
        **data.model_fields,
    }
    if args.model_max_seq_len is not None:
        spec["max_seq_len"] = max(int(spec["max_seq_len"]), args.model_max_seq_len)
    return spec


def _to_device(batch: Any, device: torch.device, task_type: str) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    if task_type == "regression":
        inputs, targets = batch
        return inputs.to(device), targets.to(device), None, None
    inputs = batch["inputs"].to(device)
    targets = batch["targets"].to(device)
    mask = batch["loss_mask"].to(device)
    metadata = batch["metadata"].to(device)
    return inputs, targets, mask, metadata


def compute_batch(
    model: nn.Module,
    batch: Any,
    device: torch.device,
    task_type: str,
) -> tuple[Tensor, dict[str, float]]:
    inputs, targets, mask, _ = _to_device(batch, device, task_type)
    predictions = model(inputs)
    if task_type == "regression":
        loss = nn.functional.mse_loss(predictions, targets)
        mae = nn.functional.l1_loss(predictions, targets)
        return loss, {"mse": float(loss.detach()), "mae": float(mae.detach())}
    loss, accuracy = masked_language_loss(predictions, targets, mask)
    return loss, {
        "cross_entropy": float(loss.detach()),
        "accuracy": float(accuracy.detach()),
        "perplexity": float(torch.exp(loss.detach().clamp(max=20.0))),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task_type: str,
    max_batches: int = 50,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        _, metrics = compute_batch(model, batch, device, task_type)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    if count == 0:
        raise RuntimeError("Validation loader produced no batches.")
    return {key: value / count for key, value in totals.items()}


def support_regime_purity(weights: Tensor, regimes: Tensor, mask: Tensor) -> float:
    """Compute majority-label purity of dominant persistent supports."""
    # weights: [B, H, T, M], regimes/mask: [B, T]
    dominant = weights.mean(dim=1).argmax(dim=-1)
    valid = (mask > 0) & (regimes >= 0)
    if not bool(valid.any()):
        return float("nan")
    dominant_np = dominant[valid].detach().cpu().numpy()
    regimes_np = regimes[valid].detach().cpu().numpy()
    correct = 0
    for support in np.unique(dominant_np):
        labels = regimes_np[dominant_np == support]
        counts = np.bincount(labels)
        correct += int(counts.max())
    return correct / len(dominant_np)


@torch.no_grad()
def save_diagnostics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    task_type: str,
) -> dict[str, float]:
    if not isinstance(model, KAMSequenceModel):
        return {}
    model.eval()
    batch = next(iter(loader))
    inputs, _, mask, metadata = _to_device(batch, device, task_type)
    _, diagnostics = model(inputs, return_weights=True)
    payload: dict[str, np.ndarray] = {}
    metrics: dict[str, float] = {}

    if diagnostics.context_weights:
        context = diagnostics.context_weights[-1][0].detach().cpu().numpy()
        payload["context_weights"] = context
        plt.figure(figsize=(5.2, 4.2))
        plt.imshow(context.mean(axis=0), aspect="auto", origin="upper")
        plt.xlabel("key position")
        plt.ylabel("query position")
        plt.title("Mean context attention")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(output_dir / "context_attention.png", dpi=160)
        plt.close()

    if diagnostics.memory_weights:
        memory = diagnostics.memory_weights[-1][0].detach().cpu().numpy()
        payload["memory_weights"] = memory
        plt.figure(figsize=(5.2, 4.2))
        plt.imshow(memory.mean(axis=0), aspect="auto", origin="upper")
        plt.xlabel("persistent support")
        plt.ylabel("token position")
        plt.title("Mean persistent-memory attention")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(output_dir / "memory_attention.png", dpi=160)
        plt.close()
        if metadata is not None and mask is not None and metadata.ndim == 2:
            metrics["support_regime_purity"] = support_regime_purity(
                diagnostics.memory_weights[-1], metadata, mask
            )

    if payload:
        np.savez_compressed(output_dir / "attention_diagnostics.npz", **payload)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a KAM experiment or matched baseline.")
    parser.add_argument("--task", choices=["mackey-glass", "copy", "regime", "char"], required=True)
    parser.add_argument(
        "--model",
        choices=["kam", "kernel-self", "memory-only", "dot-transformer", "dot-hybrid", "gru", "mlp"],
        default="kam",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/run"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-supports", type=int, default=64)
    parser.add_argument("--context-window", type=int, default=None)
    parser.add_argument("--model-max-seq-len", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--hide-memory-weights", action="store_true")

    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--series-length", type=int, default=12000)
    parser.add_argument("--tau", type=float, default=17.0)
    parser.add_argument("--beta", type=float, default=0.2)

    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--copy-length", type=int, default=16)
    parser.add_argument("--alphabet-size", type=int, default=17)
    parser.add_argument("--num-regimes", type=int, default=4)
    parser.add_argument("--regime-switch-validation", action="store_true")

    default_text = Path(__file__).resolve().parent / "data" / "sample_text.txt"
    parser.add_argument("--text-path", type=Path, default=default_text)
    parser.add_argument("--char-stride", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    data = build_data(args)
    model_spec = build_model_spec(args, data)
    model = make_model(model_spec).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    history: list[dict[str, Any]] = []
    best_validation = math.inf
    def infinite_batches(loader: DataLoader):
        while True:
            yield from loader

    training_iterator = infinite_batches(data.train_loader)
    start_time = time.perf_counter()

    for step in range(1, args.steps + 1):
        model.train()
        batch = next(training_iterator)
        optimizer.zero_grad(set_to_none=True)
        loss, train_metrics = compute_batch(model, batch, device, data.task_type)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            validation = evaluate(
                model, data.validation_loader, device, data.task_type, args.eval_batches
            )
            primary = validation["mse"] if data.task_type == "regression" else validation["cross_entropy"]
            record = {
                "step": step,
                "train": train_metrics,
                "validation": validation,
                "elapsed_seconds": time.perf_counter() - start_time,
            }
            history.append(record)
            print(record, flush=True)
            if primary < best_validation:
                best_validation = primary
                atomic_torch_save(
                    args.output / "best_model.pt",
                    {
                        "model_state": model.state_dict(),
                        "model_spec": model_spec,
                        "task": args.task,
                        "data_metadata": data.metadata,
                        "args": vars(args),
                        "validation": validation,
                    },
                )

    checkpoint = torch.load(args.output / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    final_validation = evaluate(model, data.validation_loader, device, data.task_type, args.eval_batches)
    diagnostic_metrics = save_diagnostics(
        model, data.validation_loader, device, args.output, data.task_type
    )
    summary = {
        "task": args.task,
        "model": args.model,
        "device": str(device),
        "parameter_count": parameter_count,
        "model_spec": model_spec,
        "data_metadata": data.metadata,
        "best_validation": checkpoint["validation"],
        "final_validation": final_validation,
        "diagnostics": diagnostic_metrics,
        "history": history,
        "total_seconds": time.perf_counter() - start_time,
    }
    save_json(args.output / "metrics.json", summary)
    print(f"Saved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
