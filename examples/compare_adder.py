"""Train an adder root and compare its learning curve with a released root."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from synaptic_lens_target_model_factory.arithmetic.syn88 import (
    AGES,
    RULES,
    generate_root,
    root_lineage,
)


def read_reference(path, sha256):
    with Path(path).open("rb") as stream:
        data = stream.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024 or hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError("Reference size or SHA-256 mismatch")
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    if (
        payload["weights"].shape != (2, 8, 1299)
        or payload["addition_correct"].shape != (2, 8, 100)
        or not np.array_equal(payload["ages"], AGES)
        or not np.isfinite(payload["weights"]).all()
        or not np.isin(payload["addition_correct"], [0, 1]).all()
    ):
        raise ValueError("Expected a SYN88 paired target root")
    index = int(payload["root_index"])
    if not 0 <= index < 2304 or int(payload["root_seed"]) != 492232 + index:
        raise ValueError("Invalid root identity")
    return payload


def compare(reference, trained):
    return [
        dict(
            rule=rule,
            updates=age,
            reference_addition_accuracy=float(
                reference["addition_correct"][r, a].mean()
            ),
            trained_addition_accuracy=float(trained["addition_correct"][r, a].mean()),
        )
        for r, rule in enumerate(RULES)
        for a, age in enumerate(AGES)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu", help="default: cpu"
    )
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error(
            "CUDA is unavailable. Install a CUDA PyTorch build or select --device cpu."
        )
    if args.output.exists():
        parser.error("Choose a new output directory; existing results are preserved.")
    reference = read_reference(args.reference, args.sha256)
    torch.set_num_threads(1)
    trained = generate_root(int(reference["root_index"]), device=args.device)
    rows = compare(reference, trained)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "target.npz").open("xb") as stream:
        np.savez_compressed(stream, **trained)
    with (args.output / "comparison.csv").open(
        "x", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(
        device=args.device,
        reference_sha256=args.sha256,
        training=root_lineage(trained),
        comparison=rows,
    )
    (args.output / "comparison.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("rule       updates  released  yours (addition accuracy)")
    for row in rows:
        print(
            f"{row['rule']:10} {row['updates']:7}  {row['reference_addition_accuracy']:7.1%}  {row['trained_addition_accuracy']:7.1%}"
        )
    print(f"Saved target and comparison to {args.output}")


if __name__ == "__main__":
    main()
