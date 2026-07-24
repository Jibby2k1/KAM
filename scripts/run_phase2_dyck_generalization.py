from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from kam.data import BoundedDyck2Dataset
from kam.factory import make_model


def valid_brackets(tokens: list[int]) -> bool:
    pairs = {2: 1, 4: 3}
    stack: list[int] = []
    for token in tokens:
        if token in {1, 3}:
            stack.append(token)
        elif token in pairs:
            if not stack or stack.pop() != pairs[token]:
                return False
        else:
            return False
    return not stack


@torch.no_grad()
def evaluate(model, depth: int, device: torch.device, examples: int, batch_size: int) -> dict[str, float]:
    dataset = BoundedDyck2Dataset(size=examples, max_depth=depth, min_depth=depth, seed=100 + depth)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = total = valid = 0
    model.eval()
    for batch in loader:
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
        logits = model(inputs)
        predicted = logits.argmax(dim=-1)
        correct += int((predicted == targets).sum())
        total += int(targets.numel())
        valid += sum(valid_brackets(row.tolist()) for row in predicted.cpu())
    return {"depth": float(depth), "token_accuracy": correct / max(total, 1), "grammar_valid_fraction": valid / max(examples, 1), "examples": float(examples)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Dyck-2 depth generalization.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/phase2/dyck_generalization.csv"))
    parser.add_argument("--depths", type=int, nargs="+", default=[8, 10, 12, 16])
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = make_model(checkpoint["model_spec"]).to(args.device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    rows = [evaluate(model, depth, torch.device(args.device), args.examples, args.batch_size) for depth in args.depths]
    for row in rows:
        row.update({"checkpoint": str(args.checkpoint), "variant": checkpoint["model_spec"].get("model_name", "unknown"), "position_mode": checkpoint["model_spec"].get("position_mode", "learned")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} Dyck generalization rows to {args.output}")


if __name__ == "__main__":
    main()
