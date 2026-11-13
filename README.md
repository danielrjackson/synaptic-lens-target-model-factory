# Synaptic Lens Target Model Factory

Author-developed recipes for target networks studied by [Synaptic Lens](https://github.com/danielrjackson/synaptic-lens).
Targets solve arithmetic or image tasks; Lens observers predict properties of
those targets and live in the companion repository. These are separate models.

## Install and try a bounded CPU example

Use Python 3.13 or later. From a checkout, create a virtual environment and
install a platform-compatible PyTorch build, then this package:

```console
python3.13 -m venv .venv
.venv/bin/python -m pip install -e ".[test]"
.venv/bin/python examples/train_vision.py --family shapes
.venv/bin/python -m pytest tests -q
```

On native Windows no activation is necessary:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe examples/train_vision.py --family sprites
.\.venv\Scripts\python.exe -m pytest tests -q
```

See the [official PyTorch installation guide](https://pytorch.org/get-started/locally/)
for CPU-specific wheels. Examples use CPU only and fresh temporary directories.
They do not download MNIST or train a historical population.

## Arithmetic targets

The SYN88 recipe is available as:

```console
python -m synaptic_lens_target_model_factory.arithmetic.syn88 ./new-targets --count 1
```

It trains a 20-input, 32-hidden-ReLU, 19-output MLP on all 100 ordered digit pairs.
Each root has addition and alternate-rule trajectories with identical birth
weights, eight ages and seed `492232 + root_index`. Root splits use seed 880088:
1,536 TRAIN, 256 validation, 512 historical test roots. The 2,304 roots contain
36,864 states, with shared paired births; checkpoints are not independent roots.
Numeric NPZ exports omit optimizer/RNG state. New generation is not promised to
reproduce historical bytes across PyTorch/device versions. Existing outputs are
never replaced. The default generates one CPU root, not the full population.

This factory is the intended canonical home for future generator work. Its first
SYN88 implementation is unchanged from the existing public SL generator; SL's
currently usable compatibility copy is frozen during the reviewed cross-repo
transition. The compatibility import change must wait for an installable public
factory revision. No divergent algorithm is introduced here.

[Arithmetic species](docs/ARITHMETIC.md) inventories twelve exact historical label
maps. Their labels train targets; Lens supervision may still ask about addition.

## Vision targets and data rights

The package includes verified d32-L2 (19,658 parameters), d64-L2 (72,074) and
d64-L4 (139,018) patch-transformer definitions. `vision.model` retains the
root-explicit d32 recipe; `vision.sizes` retains config-specific initialization
seeds. These conventions are distinct and must not be interchanged silently.
[Vision recipes](docs/VISION.md) cover ordinary MNIST, contrarian labels, and six
separate synthetic families: sprites, shapes, textures, counting, location and
arrows. Broader vision is not all MNIST. These classifiers naturally consume
images; no Lens observer receiving individual task images is included.

No raw MNIST images or external pretrained weights are bundled. MNIST is by Yann
LeCun, Corinna Cortes and Christopher J. C. Burges; obtain it upstream under its
own terms. [Upstream torchvision resource metadata](https://raw.githubusercontent.com/pytorch/vision/main/torchvision/datasets/mnist.py)
identifies download checksums; it does not grant dataset redistribution rights.

Archived byte-exact initialization is **not established** on Windows. The same
three states are generated under both tested PyTorch 2.14 and 2.15 CPU profiles,
and both differ from the archived digests. A small published birth-state fixture
isolates uniform-affine FP32 rounding differences and a two-ULP positional-token
residual. See [compatibility evidence](docs/initialization-compatibility.json).
`vision.compatibility.compare_initializer` hash-binds a supplied reference and
reports byte-level and ULP differences without changing either state. Current
seeded repeatability and strict checkpoint loading are separate tested contracts.

## Artifacts, evidence and citation

Weights live on Hugging Face, not in code Git. [Release identities](docs/releases.json)
pin manifests for the arithmetic and MNIST families. Existing model cards and
[SL milestones](https://github.com/danielrjackson/synaptic-lens/tree/main/docs/milestones)
state what was measured and preserve negative results. This package distributes
recipes, not new research conclusions. Sealed SYN53 holdouts, operational ledgers
and unresolved later vision-study claims are excluded.

Daniel Jackson develops this research; AI tools assisted engineering and curation.
Cite this repository and the revision used via [CITATION.cff](CITATION.cff).
Source uses unmodified [AI Pubs research-use RAIL-S 0.1, March 2, 2023](LICENSE.md).
The companion [RAIL-M](LICENSE-MODEL.md) applies only to model files explicitly
designated by a release, not datasets or third-party material. These are research
use restrictions, not OSI open-source licenses or general noncommercial terms.
Retain [NOTICE](NOTICE); dependencies and datasets retain their own rights.
