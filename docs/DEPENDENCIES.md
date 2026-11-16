# Current Windows CUDA and CPU environments

Verified package-index versions on 2026-11-16: Python 3.15.2, uv 0.13.8,
PyTorch 2.15.0, CUDA wheel runtime 13.4, NumPy 2.5.5, Pillow 13.0.0,
pytest 9.2.0 and setuptools 84.2.1. Every runtime/test/build transitive dependency
is pinned in `requirements/windows-{cuda,cpu}.in` and archive-hashed in its `.txt`.
SymPy 1.14.0 requires mpmath <1.4, so mpmath 1.3.0 is the newest compatible
release despite the newer 1.4.1 release. Historical scientific evidence stays
unchanged; these profiles establish current execution and seeded repeatability.

## Install and verify

Install official Windows x64 Python 3.15.2 first. In PowerShell:

```powershell
if (Test-Path .venv-installer) { throw "Choose a new installer environment path" }
py -3.15 -m venv .venv-installer
.\.venv-installer\Scripts\python.exe -m pip install --require-hashes --only-binary=:all: -r requirements/installer.txt
.\.venv-installer\Scripts\python.exe scripts/locked.py install
.\.venv-installer\Scripts\python.exe scripts/locked.py test
```

CUDA is the default profile. It installs PyTorch's official `cu134` Windows
wheel and requires a compatible NVIDIA driver/GPU. The wheel supplies its CUDA
runtime; installing a separate CUDA toolkit is unnecessary for these Python
recipes. `test` requires real CUDA availability and executes a small GPU tensor
operation before the installed-package suite. Training CLIs default to CUDA.

For CPU, add `--profile cpu` to both locked commands and `--device cpu` to training
examples. CPU execution explicitly hides GPUs. The manual-only Windows CI uses
this CPU profile because the hosted runner has no NVIDIA GPU. CPU and CUDA use
separate fresh directories; an existing environment is never overwritten. Use
`--environment PATH` for another fresh destination.

The bootstrap remains isolated from shared Python installations. The installer
uses hash-required, wheel-only synchronization, builds this checkout with the
locked setuptools backend, and installs its local wheel without dependency
resolution. Tests validate installed versions and package location. The Python
distribution and initial pip remain explicit bootstrap trust inputs.

## Refreshing locks

Use uv 0.13.8 and deliberately reviewed pins:

```powershell
uv pip compile requirements/windows-cpu.in --python-version 3.15 --python-platform windows --generate-hashes --no-build --no-header --no-annotate -o requirements/windows-cpu.txt
uv pip compile requirements/windows-cuda.in --python-version 3.15 --python-platform windows --generate-hashes --no-build --no-header --no-annotate --index https://download.pytorch.org/whl/cu134 --index-strategy unsafe-best-match -o requirements/windows-cuda.txt
uv pip compile requirements/installer.in --python-version 3.15 --python-platform windows --generate-hashes --no-build --no-header --no-annotate -o requirements/installer.txt
```

The CUDA resolution deliberately considers both official PyPI and PyTorch indexes:
PyTorch's index also mirrors older common dependencies. The explicit `+cu134`
torch version and resulting SHA-256 hashes bind the CUDA build; all installations
require those hashes. Review new archive identities before accepting a lock.
Refresh `integrity.json` hashes for input/output bytes and LF-normalized pyproject.
Requirement/manifest files use LF; maintained Windows source/docs use CRLF.
Run clean installs for both profiles, bounded GPU verification, all tests, and
an independent exact-diff review. Shared environments and historical artifacts
are preserved.

## Official release sources

- https://www.python.org/downloads/
- https://pypi.org/project/uv/
- https://pytorch.org/get-started/locally/
- https://download.pytorch.org/whl/cu134/torch/
- https://pypi.org/project/numpy/
- https://pypi.org/project/sympy/
