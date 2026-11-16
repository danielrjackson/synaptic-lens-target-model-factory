import numpy as np
import pytest

from synaptic_lens_target_model_factory.vision.generators import (  # noqa: E402
    CLASS_CODEBOOKS,
    _arrow_raster_direction,
    _arrow_rotation_degrees,
    _count_masks,
    _crop_mirror_keys,
    _location_component_sizes,
    _sword_raster_measurements,
    _valid_key_raster,
    _valid_key_sprite,
    _valid_mushroom_sprite,
    _valid_shield_sprite,
    _valid_star_sprite,
    _valid_sword_raster,
    _valid_sword_sprite,
    generate_arrow_dataset,
    generate_balanced_dataset,
    generate_count_counterfactuals,
    generate_sprite_designs,
    generate_sprite_examples,
)


@pytest.mark.parametrize(
    "family", ("shapes", "textures", "count", "location", "arrows")
)
def test_species_generators_are_balanced_seeded_and_varied(family):
    images, labels = generate_balanced_dataset(
        family, examples_per_class=16, seed=70101
    )
    repeat, repeat_labels = generate_balanced_dataset(
        family, examples_per_class=16, seed=70101
    )
    other, _ = generate_balanced_dataset(family, examples_per_class=16, seed=70103)
    assert images.shape == (160, 28, 28) and images.dtype == np.uint8
    assert labels.shape == (160,) and labels.dtype == np.uint8
    assert np.bincount(labels, minlength=10).tolist() == [16] * 10
    assert np.array_equal(images, repeat) and np.array_equal(labels, repeat_labels)
    assert not np.array_equal(images, other)
    for label in range(10):
        assert np.unique(images[labels == label].reshape(16, -1), axis=0).shape[0] >= 10


def test_textures_have_exactly_matched_binary_ink():
    images, _ = generate_balanced_dataset("textures", examples_per_class=10, seed=70201)
    assert np.all(np.sum(images == 255, axis=(1, 2)) == 392)
    assert set(np.unique(images).tolist()) == {0, 255}


def _component_degrees(mask):
    coords = {tuple(point) for point in np.argwhere(mask)}
    edges = set()
    for y, x in coords:
        for neighbor in ((y + 1, x), (y, x + 1)):
            if neighbor in coords:
                edges.add(tuple(sorted(((y, x), neighbor))))
    adjacency = {p: set() for p in coords}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    visited = set()
    stack = [next(iter(coords))]
    while stack:
        point = stack.pop()
        if point not in visited:
            visited.add(point)
            stack.extend(adjacency[point] - visited)
    return len(coords), len(edges), len(visited)


@pytest.mark.parametrize("ink", (8, 10, 12))
def test_count_target_is_closed_loop_and_distractor_is_equal_ink_tree(ink):
    loop, tree = _count_masks(ink)
    ln, le, lc = _component_degrees(loop)
    tn, te, tc = _component_degrees(tree)
    assert (ln, lc, le) == (ink, ink, ink)
    assert (tn, tc, te) == (ink, ink, ink - 1)


def test_count_total_ink_is_independent_of_target_count():
    images, labels = generate_balanced_dataset(
        "count", examples_per_class=60, seed=70301
    )
    ink = np.sum(images == 255, axis=(1, 2))
    assert set(ink.tolist()) == {72, 90, 108}
    for label in range(10):
        assert set(ink[labels == label].tolist()) == {72, 90, 108}


def test_count_counterfactuals_change_one_equal_ink_motif_only():
    images, labels, changed_sites = generate_count_counterfactuals(seed=9500003)
    assert images.shape == (900, 2, 28, 28) and labels.shape == (900, 2)
    assert np.bincount(labels[:, 0], minlength=9).tolist() == [100] * 9
    assert np.array_equal(labels[:, 1], labels[:, 0] + 1)
    assert np.array_equal(
        np.sum(images[:, 0] == 255, axis=(1, 2)),
        np.sum(images[:, 1] == 255, axis=(1, 2)),
    )
    for (before, after), site in zip(images, changed_sites):
        row, col = divmod(int(site), 3)
        y0, x0 = 2 + 10 * row, 2 + 10 * col
        patch = np.zeros((28, 28), dtype=bool)
        patch[y0 : y0 + 5, x0 : x0 + 5] = True
        difference = before != after
        assert np.any(difference & patch)
        assert not np.any(difference & ~patch)


def test_location_centers_and_edges_are_valid():
    images, labels = generate_balanced_dataset(
        "location", examples_per_class=256, seed=70401
    )
    for image, label in zip(images, labels):
        y, x = np.nonzero(image >= 200)
        cx, cy = x.mean(), y.mean()
        observed = 5 * int(cy // 14) + int(cx // 5.6)
        assert observed == int(label)
        assert cx % 5.6 >= 1.0 and 5.6 - cx % 5.6 >= 1.0
        assert cy % 14 >= 1.0 and 14 - cy % 14 >= 1.0
        assert np.all((x // 5.6).astype(int) == int(label) % 5)
        assert np.all((y // 14).astype(int) == int(label) // 5)
        faint = image[(image >= 30) & (image < 200)]
        assert faint.size > 0 and int(faint.max()) <= 90
        assert len(_location_component_sizes((image >= 30) & (image < 200))) == 3
        assert min(_location_component_sizes((image >= 30) & (image < 200))) >= 4
        bright = image >= 200
        bright_y, bright_x = np.nonzero(bright)
        faint_y, faint_x = np.nonzero((image >= 30) & (image < 200))
        distance = np.maximum(
            np.abs(faint_y[:, None] - bright_y[None, :]),
            np.abs(faint_x[:, None] - bright_x[None, :]),
        )
        assert int(distance.min()) >= 2
        assert x.min() >= 0 and x.max() <= 27 and y.min() >= 0 and y.max() <= 27
    for label in range(10):
        positions = []
        for image in images[labels == label]:
            y, x = np.nonzero(image >= 200)
            positions.append((round(float(x.mean()), 3), round(float(y.mean()), 3)))
        assert len(set(positions)) >= 4


def test_arrow_bins_keep_four_degree_boundary_margin():
    for label in range(10):
        for within in (4.0, 18.0, 32.0):
            rotation = _arrow_rotation_degrees(label, within)
            assert abs(rotation + 36 * label + within) < 1e-9
    with pytest.raises(ValueError):
        _arrow_rotation_degrees(0, 3.9)
    with pytest.raises(ValueError):
        _arrow_rotation_degrees(0, 32.1)


def test_arrow_full_raster_direction_matches_bins_and_target_angle_margin():
    images, labels, target_angles = generate_arrow_dataset(
        examples_per_class=128, seed=70403
    )
    generic_images, generic_labels = generate_balanced_dataset(
        "arrows", examples_per_class=128, seed=70403
    )
    assert np.array_equal(images, generic_images) and np.array_equal(
        labels, generic_labels
    )
    bin_starts = labels.astype(np.float32) * 36
    assert np.all(target_angles >= bin_starts + 4)
    assert np.all(target_angles <= bin_starts + 32)
    inferred = np.asarray(
        [_arrow_raster_direction(image)[0] for image in images], dtype=np.uint8
    )
    assert np.array_equal(inferred, labels)


def test_sprite_designs_are_whole_split_disjoint_under_crop_and_mirror():
    train_x, train_y, val_x, val_y = generate_sprite_designs(
        train_seed=70501,
        validation_seed=70503,
        train_designs_per_class=16,
        validation_designs_per_class=8,
    )
    assert train_x.shape == (160, 12, 12) and val_x.shape == (80, 12, 12)
    assert np.bincount(train_y, minlength=10).tolist() == [16] * 10
    assert np.bincount(val_y, minlength=10).tolist() == [8] * 10
    for label in range(10):
        train_keys = set().union(
            *(_crop_mirror_keys(x) for x in train_x[train_y == label])
        )
        val_keys = set().union(*(_crop_mirror_keys(x) for x in val_x[val_y == label]))
        assert not train_keys.intersection(val_keys)
    feature_audit = {}
    examples, labels = generate_sprite_examples(
        train_x, train_y, examples_per_class=20, seed=70505, feature_audit=feature_audit
    )
    assert examples.shape == (200, 28, 28) and examples.dtype == np.uint8
    assert np.bincount(labels, minlength=10).tolist() == [20] * 10
    assert np.all(np.any(examples > 0, axis=(1, 2)))
    assert all(_valid_key_raster(image) for image in examples[labels == 2])
    assert all(_valid_sword_raster(image) for image in examples[labels == 4])
    assert feature_audit["key"]["rows_checked"] == 20
    assert feature_audit["key"]["rows_with_visible_tooth"] == 20
    assert feature_audit["key"]["minimum_visible_tooth_count"] >= 1
    assert feature_audit["key"]["minimum_visible_tooth_depth_px"] >= 2
    assert feature_audit["sword"]["rows_checked"] == 20
    assert feature_audit["sword"]["minimum_blade_to_hilt_height_ratio"] >= 2
    assert feature_audit["sword"]["maximum_pommel_overhang_px"] <= 1
    assert feature_audit["sword"]["maximum_hilt_height_px"] <= 4
    assert all(
        _sword_raster_measurements(image)["pommel_overhang_px"] == 0
        for image in examples[labels == 4]
    )
    # Every canonical foreground pixel is expanded to a 2x2 block for all labels.
    assert np.all(np.count_nonzero(examples, axis=(1, 2)) % 4 == 0)


def test_sprite_protocol_supports_full_per_root_design_counts():
    train_x, train_y, val_x, val_y = generate_sprite_designs(
        train_seed=9221007,
        validation_seed=9400000,
        train_designs_per_class=128,
        validation_designs_per_class=32,
    )
    assert train_x.shape == (1280, 12, 12) and val_x.shape == (320, 12, 12)
    assert np.bincount(train_y, minlength=10).tolist() == [128] * 10
    assert np.bincount(val_y, minlength=10).tolist() == [32] * 10
    assert all(_valid_key_sprite(image) for image in train_x[train_y == 2])
    assert all(_valid_key_sprite(image) for image in val_x[val_y == 2])
    assert all(_valid_sword_sprite(image) for image in train_x[train_y == 4])
    assert all(_valid_sword_sprite(image) for image in val_x[val_y == 4])
    for images, labels in ((train_x, train_y), (val_x, val_y)):
        assert all(_valid_star_sprite(image) for image in images[labels == 1])
        assert all(_valid_mushroom_sprite(image) for image in images[labels == 3])
        assert all(_valid_shield_sprite(image) for image in images[labels == 6])
    other_root, other_y, shared_val, shared_val_y = generate_sprite_designs(
        train_seed=9221017,
        validation_seed=9400000,
        train_designs_per_class=128,
        validation_designs_per_class=32,
    )
    assert np.array_equal(val_x, shared_val) and np.array_equal(val_y, shared_val_y)
    assert (
        other_root.shape == (1280, 12, 12)
        and np.bincount(other_y, minlength=10).tolist() == [128] * 10
    )


def test_all_species_codebooks_are_explicitly_ten_class():
    assert set(CLASS_CODEBOOKS) == {
        "sprites",
        "shapes",
        "textures",
        "count",
        "location",
        "arrows",
    }
    assert all(
        len(codebook) == 10 and len(set(codebook)) == 10
        for codebook in CLASS_CODEBOOKS.values()
    )
