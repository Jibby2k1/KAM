#!/usr/bin/env python3
"""Grade generated code inside an isolated Python container."""
import argparse
import ast
import json
import os
import secrets
import subprocess
import tempfile
from pathlib import Path

from run_kam_general_benchmarks import ARMS, digest, load_cases

SANDBOX = (
    "unshare", "--user", "--map-current-user", "--net", "--",
    "apptainer", "exec", "--userns", "--containall", "--cleanenv",
    "--no-mount", "hostfs,cwd,home,bind-paths", "--pwd", "/tmp",
)
ENV = {key: os.environ[key] for key in ("PATH",) if key in os.environ}


def isolated_run(image, command, script=None, timeout=12):
    binding = [] if script is None else ["--bind", f"{script}:/tmp/eval_case.py:ro"]
    return subprocess.run(
        [*SANDBOX, *binding, str(image), *command], env=ENV,
        cwd="/tmp", stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout,
    )


def check_isolation(image):
    command = (
        "import os, socket; "
        "assert not os.path.exists('/blue/bala1s/rvalle1'); "
        "assert not os.path.exists('/home/rvalle1/.ssh'); "
        "assert socket.if_nameindex() == [(1, 'lo')], socket.if_nameindex()"
    )
    result = isolated_run(image, ["python", "-I", "-c", command])
    if result.returncode:
        raise RuntimeError(f"Isolation preflight failed: {result.stderr.decode(errors='replace')}")


def get_program(task, case, completion):
    if task == "humaneval":
        return case["prompt"] + completion, case["test"] + "\ncheck(" + case["entry_point"] + ")\n"
    return completion, "\n".join(case["test_list"]) + "\n"


def evaluate(task, case, completion, image):
    program, tests = get_program(task, case, completion)
    try:
        tree = ast.parse(program)
    except (ValueError, SyntaxError):
        return "syntax_error"
    if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
               for node in tree.body):
        return "no_definition"
    nonce = secrets.token_hex(16)
    setup = case["test_setup_code"] or "" if task == "mbpp" else ""
    # Keep the resource limits in the trusted wrapper and compile the program
    # separately so appending the tests cannot repair an incomplete definition.
    source = f"""import resource
resource.setrlimit(resource.RLIMIT_AS, (384 << 20, 384 << 20))
resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
scope = {{"__name__": "__main__"}}
exec(compile({setup!r}, "<setup>", "exec"), scope)
exec(compile({program!r}, "<solution>", "exec"), scope)
exec(compile({tests!r}, "<tests>", "exec"), scope)
print({nonce!r})
"""
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        script = Path(directory) / "case.py"
        script.write_text(source)
        try:
            result = isolated_run(image, ["python", "-I", "/tmp/eval_case.py"], script)
        except subprocess.TimeoutExpired:
            return "timeout"
    if b"FATAL:" in result.stderr and b"/tmp/eval_case.py" not in result.stderr:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    return "passed" if result.returncode == 0 and nonce.encode() in result.stdout.splitlines() else "failed_tests"


def main(args):
    check_isolation(args.image)
    source, cases = load_cases(args.data, args.task)
    example = cases[0]
    canonical = example["canonical_solution"] if args.task == "humaneval" else example["code"]
    assert evaluate(args.task, example, canonical, args.image) == "passed", "reference grader control"
    if args.preflight:
        print(f"{args.task} isolated reference control passed")
        return
    generation = json.loads(args.generation.read_text())
    metadata = generation["metadata"]
    assert metadata["task"] == args.task and metadata["arm"] in ARMS
    assert digest(source) == metadata["source_sha256"]
    rows = []
    for index, row in enumerate(generation["records"]):
        assert index == row["index"]
        outcome = evaluate(args.task, cases[index], row["completion"], args.image)
        rows.append({"index": index, "prompt_bytes": row["prompt_bytes"],
                     "correct": outcome == "passed", "outcome": outcome})
    assert len(rows) == len(cases)
    metadata["sandbox_sha256"] = digest(args.image)
    args.output.write_text(json.dumps({"metadata": metadata, "records": rows}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("humaneval", "mbpp"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--generation", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    main(parser.parse_args())
