"""One synthetic vision-family target update on CUDA (or explicit CPU); no MNIST download."""

import argparse
import json
import tempfile
from pathlib import Path

import torch

from synaptic_lens_target_model_factory.vision.generators import (
    generate_balanced_dataset,
    generate_sprite_designs,
    generate_sprite_examples,
)
from synaptic_lens_target_model_factory.vision.recipes import (
    load_target,
    save_target,
    train_fixture,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        choices=["sprites", "shapes", "textures", "count", "location", "arrows"],
        default="shapes",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.family == "sprites":
        designs, labels, _, _ = generate_sprite_designs(
            train_seed=1,
            validation_seed=2,
            train_designs_per_class=2,
            validation_designs_per_class=1,
        )
        x, y = generate_sprite_examples(designs, labels, examples_per_class=1, seed=3)
    else:
        x, y = generate_balanced_dataset(args.family, examples_per_class=1, seed=3)
    model, lineage = train_fixture(x, y, device=args.device)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "target.pt"
        digest = save_target(path, model, lineage)
        load_target(path, digest)
    print(
        json.dumps(
            {
                "family": args.family,
                "scope": "synthetic smoke, not historical collection reproduction",
                **lineage,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
