# Train an adder and compare it with the released example

This walkthrough downloads one trained target, trains the same architecture and
seed on your machine, and compares both learning curves. It takes 200 small
optimizer steps: 100 for addition and 100 for the alternate arithmetic rule.

## 1. Prepare the environment

Install the [locked Windows CUDA profile](DEPENDENCIES.md#optional-cuda-profile)
from this checkout. For CPU, use the default locked profile instead and
replace `.venv-locked-cuda` with `.venv-locked` in the commands below. Keep
the reference download in a small separate environment, as in the
[download guide](DOWNLOADS.md):

```powershell
py -3.13 -m venv .venv-hub
.\.venv-hub\Scripts\python.exe -m pip install "huggingface_hub==2.4.0"
$download = .\.venv-hub\Scripts\python.exe scripts/download_models.py --release synaptic-lens-targets-adder --file syn88/population/root-0000.npz | ConvertFrom-Json
$reference = $download.files.'syn88/population/root-0000.npz'
```

The helper checks the release manifest, checkpoint size, and SHA-256. It reuses the
Hugging Face cache on later runs. For other shells, use the equivalent interpreter path and copy the returned
`path` and `sha256` into the next command. The pinned environments above
are validated only on Windows.

## 2. Train and compare

```powershell
.\.venv-locked-cuda\Scripts\python.exe examples/compare_adder.py --reference $reference.path --sha256 $reference.sha256 --output runs/adder-01 --device cuda
```

For a CPU-only machine, pass `--device cpu`. Choose a new output directory each
time. The command preserves the download and all existing runs.

You will see a table with released and newly trained addition accuracy at updates
0, 1, 2, 5, 10, 25, 50 and 100. Both models are evaluated on all 100 ordered digit
pairs. The alternate-rule rows also measure **addition** accuracy: that model is
trained to answer `(2*a + b + 1) % 19`, so lower addition accuracy is expected.
These are training-task curves, rather than a held-out generalization score.

The output folder contains:

- `target.npz`: your trained weights and scores at every saved update.
- `comparison.csv`: the learning curves, ready to open in a spreadsheet or plot.
- `comparison.json`: the curves, reference digest, training seed, parameter hashes,
  device, and library versions.

Compare the learning curves first. CUDA kernels and library versions can produce
different individual weights from the archived run. You can use your new target
with the companion [Synaptic Lens examples](https://github.com/danielrjackson/synaptic-lens).

## 3. Try custom image tasks

```powershell
.\.venv-locked-cuda\Scripts\python.exe examples/train_vision.py --family shapes --device cuda
```

Choose `sprites`, `shapes`, `textures`, `count`, `location`, or `arrows`. This quick
installation check makes ten examples, runs one update of a 19,658-parameter
image transformer, and verifies a saved checkpoint before removing its temporary
files. The [vision recipes](VISION.md) describe datasets and the full training
schedule. Downloaded MNIST targets have their own release and training recipe;
this small synthetic check is a separate demonstration.
