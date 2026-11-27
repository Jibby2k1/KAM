"""Fixed registry of HiPerGator backbone and adapter experiments."""
import argparse
import hashlib
from pathlib import Path

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
REVISION = "31b70e2e869a7173562077fd711b654946d38674"
LLAMA_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
LLAMA_REVISION = "uf-hpg-2024-04-19"
LLAMA_SOURCE = Path("/data/ai/models/nlp/llama/models_llama3/Meta-Llama-3-8B-Instruct-hf")
MODEL_REGISTRY = {
    MODEL_ID: (REVISION, "smollm2-preservation-v1"),
    LLAMA_ID: (LLAMA_REVISION, "llama3-preservation-v1"),
}
SEEDS = (731201, 731211, 731221)
ARMS = ("kam", "dot")
RECIPES = {"reference": (8e-4, 0.0), "lower_lr": (8e-5, 0.0), "lower_lr_kl": (8e-5, 4.0)}
TRAIN_RUNS = tuple((seed, recipe, arm) for seed in SEEDS for recipe in RECIPES for arm in ARMS)
RUNS = (*TRAIN_RUNS, (0, "untouched", "pretrained"))
PROTOCOL = "smollm2-preservation-v1"
FACTORIAL_RECIPES = {**RECIPES, "high_lr_kl": (8e-4, 4.0)}
FACTORIAL_TRAIN_RUNS = (*TRAIN_RUNS, *((seed, "high_lr_kl", arm) for seed in SEEDS for arm in ARMS))
FACTORIAL_RUNS = (*FACTORIAL_TRAIN_RUNS, RUNS[-1])


def key(seed, recipe, arm):
    return f"{seed}-{recipe}-{arm}"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def layers_for(model):
    n = model.config.num_hidden_layers
    return tuple((i * n // 4) - 1 for i in range(1, 5))


def model_identity(root):
    model_id, revision = (root / "model-origin.txt").read_text().splitlines()
    expected, protocol = MODEL_REGISTRY[model_id]
    assert revision == expected
    return model_id, revision, protocol


def load_registered_model(root):
    from kam_pretrained_adapters import load_model

    model_id, _, _ = model_identity(root)
    fingerprints = {name: sha for sha, name in (
        line.split() for line in (root / "model-sha256.txt").read_text().splitlines()
    )}
    if model_id == LLAMA_ID:
        assert (root / "model").resolve() == LLAMA_SOURCE
        weight_names = {path.name for path in LLAMA_SOURCE.glob("model-*.safetensors")}
        assert len(weight_names) == 4 and "model.safetensors.index.json" in fingerprints
    else:
        weight_names = {"model.safetensors"}
    assert weight_names <= fingerprints.keys()
    for file in ("config.json", "tokenizer.json"):
        assert digest(root / "model" / file) == fingerprints[file]
    model, tokenizer = load_model(root)
    assert model.config.model_type == "llama"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--factorial", action="store_true")
    args = parser.parse_args()
    fit, evaluation = (FACTORIAL_TRAIN_RUNS, FACTORIAL_RUNS) if args.factorial else (TRAIN_RUNS, RUNS)
    print(*(fit if args.train else evaluation)[args.index])
