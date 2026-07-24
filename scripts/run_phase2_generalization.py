from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from kam.data import VariableCopyLanguageDataset, variable_copy_collate
from kam.factory import make_model


@torch.no_grad()
def evaluate_copy(model, dataset, device: torch.device, batch_size: int) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=variable_copy_collate)
    total_loss = 0.0
    total_correct = 0.0
    total_tokens = 0.0
    exact = 0
    examples = 0
    model.eval()
    for batch in loader:
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
        mask = batch["loss_mask"].to(device)
        logits = model(inputs)
        token_loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").reshape_as(targets)
        predictions = logits.argmax(dim=-1)
        total_loss += float((token_loss * mask).sum())
        total_correct += float(((predictions == targets).to(mask.dtype) * mask).sum())
        total_tokens += float(mask.sum())
        exact += int(((predictions == targets) | (mask == 0)).all(dim=-1).sum())
        examples += inputs.shape[0]
    return {"length": float(dataset.max_payload_length), "masked_cross_entropy": total_loss / max(total_tokens, 1.0), "copied_token_accuracy": total_correct / max(total_tokens, 1.0), "exact_sequence_accuracy": exact / max(examples, 1), "examples": float(examples)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate variable-copy length generalization from one checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/phase2/variable_copy_generalization.csv"))
    parser.add_argument("--lengths", type=int, nargs="+", default=[80, 96, 128, 192])
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = make_model(checkpoint["model_spec"]).to(args.device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    alphabet_size = int(checkpoint["model_spec"].get("vocab_size", 12)) - 4
    rows = []
    for length in args.lengths:
        dataset = VariableCopyLanguageDataset(size=args.examples, min_payload_length=length, max_payload_length=length, alphabet_size=alphabet_size, seed=1000 + length)
        row = evaluate_copy(model, dataset, torch.device(args.device), args.batch_size)
        row.update({"checkpoint": str(args.checkpoint), "variant": checkpoint["model_spec"].get("model_name", "unknown"), "position_mode": checkpoint["model_spec"].get("position_mode", "learned")})
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} length-generalization rows to {args.output}")


if __name__ == "__main__":
    main()
