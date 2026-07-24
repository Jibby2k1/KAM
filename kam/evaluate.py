from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import CopyLanguageDataset, RegimeGrammarDataset
from .factory import make_model
from .train import evaluate, save_diagnostics
from .utils import choose_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved symbolic-language checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--size", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--copy-length", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--switch-halfway", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    task = checkpoint.get("task")
    metadata = checkpoint.get("data_metadata", {})
    model = make_model(checkpoint["model_spec"]).to(device)
    model.load_state_dict(checkpoint["model_state"])

    if task == "copy":
        copy_length = args.copy_length or int(metadata["copy_length"])
        dataset = CopyLanguageDataset(
            size=args.size,
            payload_length=copy_length,
            alphabet_size=int(metadata["alphabet_size"]),
            seed=args.seed,
        )
        evaluation_name = f"copy_length_{copy_length}"
    elif task == "regime":
        sequence_length = args.seq_len or int(metadata["sequence_length"])
        dataset = RegimeGrammarDataset(
            size=args.size,
            sequence_length=sequence_length,
            alphabet_size=int(metadata["alphabet_size"]),
            num_regimes=int(metadata["num_regimes"]),
            seed=args.seed,
            switch_halfway=args.switch_halfway,
        )
        evaluation_name = f"regime_length_{sequence_length}_switch_{args.switch_halfway}"
    else:
        raise ValueError("kam-eval currently supports copy and regime checkpoints.")

    if dataset[0]["inputs"].shape[0] > int(checkpoint["model_spec"]["max_seq_len"]):
        raise ValueError(
            "Evaluation sequence exceeds the checkpoint's max_seq_len. Train with a larger "
            "--model-max-seq-len to test length generalization."
        )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    metrics = evaluate(model, loader, device, "language", max_batches=10_000)
    diagnostics = save_diagnostics(model, loader, device, args.output, "language")
    summary = {
        "checkpoint": str(args.checkpoint),
        "evaluation": evaluation_name,
        "metrics": metrics,
        "diagnostics": diagnostics,
    }
    save_json(args.output / "evaluation.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
