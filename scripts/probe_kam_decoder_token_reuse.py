#!/usr/bin/env python3
"""Compare greedy decoding with and without re-uploading the selected token."""
import argparse
import importlib.util
from pathlib import Path
from time import monotonic

import torch

from eval_kam_full_math500 import PREFIX_BUDGET
from eval_kam_pretrained import generate
from kam_adapter_preservation import digest
from kam_math_kl_protocol import write_json
from probe_kam_math500_prefix import load_panel


def measure_panels(root, work, arm, modes):
    name, model, tokenizer, stack, encoded, reference, provenance = load_panel(root, work, arm)
    first, second = modes
    panels = {}
    for size, panel in ((64, encoded), (1, encoded[:8])):
        timings = {mode: [] for mode in modes}
        expected = None
        for mode in (first, second, second, first):
            combined = []
            torch.cuda.synchronize()
            started = monotonic()
            for offset in range(0, len(panel), size):
                trace = []
                rows = modes[mode](model, stack, tokenizer, panel[offset:offset+size], "gsm8k",
                                   max_new_tokens=PREFIX_BUDGET, trace=trace, stop_on_content=False)
                combined.extend(zip(rows, trace, strict=True))
            torch.cuda.synchronize()
            timings[mode].append(monotonic() - started)
            if expected is None:
                expected = combined
            assert combined == expected, (arm, size, mode)
            if size == 64:
                assert [(text, len(tokens)) for text, (tokens, _) in combined] == reference
            print(arm, size, mode, timings[mode][-1], flush=True)
        panels[str(size)] = {"n": len(panel), "seconds": timings,
                             "generated_tokens": sum(len(tokens) for _, (tokens, _) in expected)}
    return name, provenance, panels


def probe(root, work, arm, reference_decoder):
    spec = importlib.util.spec_from_file_location("kam_reference_decoder", reference_decoder)
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    modes = {"host_upload": reference_module.generate, "device_reuse": generate}
    name, provenance, panels = measure_panels(root, work, arm, modes)
    write_json(work / f"token-transport-{name}.json", {**provenance, "panels": panels,
                "protocol": "kam-decoder-token-reuse-v1",
                "reference_decoder_sha256": digest(reference_decoder),
                "candidate_decoder_sha256": digest(Path(__file__).with_name("eval_kam_pretrained.py"))})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    cli.add_argument("work", type=Path)
    cli.add_argument("arm", choices=("kam", "dot"))
    cli.add_argument("reference_decoder", type=Path)
    args = cli.parse_args()
    probe(args.root, args.work, args.arm, args.reference_decoder)
