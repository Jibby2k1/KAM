"""Fixed registry for the HiPerGator SmolLM2 adapter preservation study."""
import argparse
import hashlib
from pathlib import Path

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
REVISION = "31b70e2e869a7173562077fd711b654946d38674"
SEEDS = (731201, 731211, 731221)
ARMS = ("kam", "dot")
RECIPES = {"reference": (8e-4, 0.0), "lower_lr": (8e-5, 0.0), "lower_lr_kl": (8e-5, 4.0)}
TRAIN_RUNS = tuple((seed, recipe, arm) for seed in SEEDS for recipe in RECIPES for arm in ARMS)
RUNS = (*TRAIN_RUNS, (0, "untouched", "pretrained"))
PROTOCOL = "smollm2-preservation-v1"


def key(seed, recipe, arm):
    return f"{seed}-{recipe}-{arm}"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def layers_for(model):
    n = model.config.num_hidden_layers
    return tuple((i * n // 4) - 1 for i in range(1, 5))


def load_registered_model(root):
    from kam_pretrained_adapters import load_model

    assert (root / "model-origin.txt").read_text().splitlines() == [MODEL_ID, REVISION]
    fingerprints = dict(line.split() for line in (root / "model-sha256.txt").read_text().splitlines())
    assert set(fingerprints.values()) == {"config.json", "tokenizer.json", "model.safetensors"}
    for file in ("config.json", "tokenizer.json"):
        assert any(name == file and digest(root / "model" / file) == sha for sha, name in fingerprints.items())
    model, tokenizer = load_model(root)
    assert model.config.model_type == "llama"
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=int)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    print(*(TRAIN_RUNS if args.train else RUNS)[args.index])
