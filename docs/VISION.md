# Image-classifier target recipes

## Ordinary MNIST and contrarian labels

Obtain official TRAIN gzip IDX files from the OSSCI mirror URLs in `vision.mnist`;
`download_train_gzip(fresh_directory)` explicitly fetches only those two TRAIN
files, verifies upstream MD5 and returns SHA-256 records. It refuses existing
directories. Alternatively obtain them yourself and use `verify_upstream_gzip`.
Decompress
locally, and parse with `parse_images`/`parse_labels`. This library bundles no
images and does not access official TEST. MD5 here matches the upstream legacy
identity, not a modern adversarial-integrity guarantee.

`make_train_validation_split` uses NumPy seed 492232. Within each class it
permutes the official TRAIN indices, selects 1,000 for validation and retains the remaining examples for
training, then sorts each combined index set. This yields disjoint 50,000/10,000
partitions. Save exact indices and hashes with each collection. Pixels become
float32 `(uint8 / 255 - 0.1307) / 0.3081`, with a singleton channel dimension.
Contrarian MNIST uses an explicitly recorded ten-class derangement; native
shifted-label accuracy and agreement with original digits are different metrics.

The historical target schedule is 100 epochs, batch size 256 (196 updates per
epoch), with 196-step linear warmup from 1e-5 to 1e-3 then cosine decay to 1e-5.
`vision.schedule.learning_rates` returns all 19,600 rates. Collection-specific
orders, initialization, checkpoints and measured labels must be recorded rather
than guessed. The public bounded `train_fixture` performs 1–10 updates on an explicit CPU or CUDA device with
AdamW at 0.001/weight decay 0.01; it is not that full historical schedule.

## Six synthetic families

- Sprites: generic heart/star/key/mushroom/sword/potion/shield/crown/flame/skull silhouettes.
- Shapes: task-native geometric classes.
- Textures: task-native patterns with matched binary ink.
- Count: object-count classes and separate counterfactual count pairs.
- Location: task-native spatial classes.
- Arrows: task-native raster direction classes.

Use `CLASS_CODEBOOKS` for exact output meanings. Output coordinate 3 does not
universally mean digit three. Independent TRAIN/validation seeds are required;
for sprites, use the whole-design split API, which excludes designs equivalent under crop/reflection across partitions. The generator tests retain semantic and
raster checks. Tiny seeded examples are not acceptance evidence for an entire
later historical collection. Semantic-repair collection acceptance, SYN93–95
and later provenance remain outside this distribution's claims.

Run `python examples/train_vision.py --family FAMILY` for each of sprites,
shapes, textures, count, location and arrows. Each generates ten synthetic
examples, performs one target update, and verifies a hash-bound restricted CPU
checkpoint roundtrip. No images go to a Lens observer.

## Architecture and reproducibility limits

The 4x4 row-major patch transformer has 50 tokens including CLS, two attention
heads, pre-norm residual blocks and an MLP feedforward width twice the model
width. d32-L2/d64-L2/d64-L4 are distinct configurations. Model tensor layouts
must match the corresponding coordinate schema. Root-explicit initialization and
config-derived run-seed initialization are separately named APIs.

See `initialization-compatibility.json`: archived initializer digests differ under both tested Windows CPU runtimes.
The same raw unit draws reproduce changed uniform tensors using fused instead
of separate FP32 affine rounding. Nineteen positional-token values in the tested
birth reference differ by at most two ULPs; that residual kernel cause remains
unresolved. A library version requirement alone does not establish byte parity. Keep source, library version, seed, task,
label mapping, data hashes, split indices, optimizer settings and actual states
in lineage manifests. Loading an existing state and recreating its initialization
are different checks.

The design uses patch-transformer mechanisms described by [Dosovitskiy et al.](https://arxiv.org/abs/2010.11929)
and [Vaswani et al.](https://arxiv.org/abs/1706.03762), with [AdamW](https://arxiv.org/abs/1711.05101).
These references explain the architecture and optimization mechanisms used.

`vision.compatibility.compare_initializer(path, sha256, initialization_seed=...)`
performs a bounded restricted CPU load and reports exact state hashes, changed
coordinates, maximum absolute differences and ULP distances. It rejects schema
or file-identity mismatches. This diagnostic reports differences while leaving both states unchanged. Preserve archived goldens unchanged.
