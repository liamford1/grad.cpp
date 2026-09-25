#!/usr/bin/env python3
"""Check that pytorch_baseline.py's preset table matches grad's.

The PyTorch baseline builds each preset's model itself, so its PRESETS
table is a second copy of src/cli/presets.cpp. This compares every field
the baseline uses against `grad presets --json` and fails on any
difference, so the head-to-head cannot silently drift into comparing
different models. Needs no torch.

Usage: python3 benchmarks/check_presets.py ./build/grad
"""

import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pytorch_baseline import LR, PRESETS  # noqa: E402  (torch is not imported)

# Baseline key -> `grad presets --json` key.
FIELDS = {
    "vocab": "vocab_size",
    "d": "d_model",
    "layers": "num_layers",
    "heads": "num_heads",
    "max_len": "max_len",
    "seq": "seq_length",
    "batch": "batch_size",
    "accum": "grad_accum",
    "dropout": "dropout",
    "modern": "modern",
}


def same(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, float) or isinstance(b, float):
        # The C++ side is float32, printed as the shortest text that reads
        # back exactly, so 0.1 arrives as 0.1.
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-12)
    return a == b


def compare(cpp: dict) -> list:
    errors = []
    if set(cpp) != set(PRESETS):
        errors.append(f"preset names differ: grad has {sorted(cpp)}, "
                      f"the baseline has {sorted(PRESETS)}")
    for name in sorted(set(cpp) & set(PRESETS)):
        ours, theirs = PRESETS[name], cpp[name]
        if set(ours) != set(FIELDS):
            errors.append(f"{name}: baseline fields {sorted(ours)} are not the checked "
                          f"set {sorted(FIELDS)}; update FIELDS in {Path(__file__).name}")
        for key, cpp_key in FIELDS.items():
            if key in ours and not same(ours[key], theirs[cpp_key]):
                errors.append(f"{name}.{key}: baseline {ours[key]!r}, "
                              f"grad {cpp_key}={theirs[cpp_key]!r}")
        # The baseline trains every preset at one learning rate.
        if not same(LR, theirs["learning_rate"]):
            errors.append(f"{name}: baseline LR {LR!r}, "
                          f"grad learning_rate={theirs['learning_rate']!r}")
    return errors


def main(argv: list) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    result = subprocess.run([argv[1], "presets", "--json"],
                            check=True, capture_output=True, text=True)
    cpp = {preset["name"]: preset for preset in json.loads(result.stdout)}
    errors = compare(cpp)
    for error in errors:
        print(f"MISMATCH {error}", file=sys.stderr)
    if errors:
        return 1
    print(f"pytorch_baseline.py presets match grad presets "
          f"({len(cpp)} presets x {len(FIELDS) + 1} fields)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
