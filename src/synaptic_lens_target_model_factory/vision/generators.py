"""Deterministic image generators for the six non-digit task species.

Species images are uint8 [N, 28, 28]. Caller-owned independent seeds must be
provided for each TRAIN and VALIDATION stream. Task codebooks are generic
answer codes; their meanings are recorded separately from model coordinates.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw

CLASS_CODEBOOKS: dict[str, tuple[str, ...]] = {
    "sprites": (
        "heart",
        "star",
        "key",
        "mushroom",
        "sword",
        "potion",
        "shield",
        "crown",
        "flame",
        "skull",
    ),
    "shapes": (
        "disk",
        "annulus",
        "elongated ellipse",
        "triangle",
        "square",
        "pentagon",
        "hexagon",
        "five point star",
        "crescent",
        "cross",
    ),
    "textures": (
        "horizontal stripes",
        "vertical stripes",
        "positive diagonal stripes",
        "negative diagonal stripes",
        "checker",
        "rings",
        "spokes",
        "dots",
        "hexagonal lattice",
        "cloud",
    ),
    "count": tuple(f"closed_loop_count_{n}" for n in range(10)),
    "location": tuple(f"region_row{n // 5}_column{n % 5}" for n in range(10)),
    "arrows": tuple(f"up_clockwise_{36 * n}_degrees" for n in range(10)),
}
DIRECTION_BINS_DEGREES = tuple(
    36 * n for n in range(10)
)  # 0 is up; clockwise in image coordinates.
_SPRITE_NAMES = CLASS_CODEBOOKS["sprites"]


class _InvalidKeySprite(ValueError):
    """A sampled key design failed its raster-level defining-feature checks."""


class _InvalidSwordSprite(ValueError):
    """A sampled sword design failed its raster-level defining-feature checks."""


class _InvalidSpriteFeatures(ValueError):
    """A sampled sprite lost a class-defining silhouette after its accents."""


def _key_hole_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return enclosed 4-connected background components for a binary sprite."""
    height, width = mask.shape
    background = ~mask.astype(bool)
    exterior: set[tuple[int, int]] = set()
    stack = [
        (y, x)
        for y in range(height)
        for x in range(width)
        if (y in (0, height - 1) or x in (0, width - 1)) and background[y, x]
    ]
    while stack:
        point = stack.pop()
        if point in exterior:
            continue
        exterior.add(point)
        y, x = point
        for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            ny, nx = neighbor
            if (
                0 <= ny < height
                and 0 <= nx < width
                and background[ny, nx]
                and neighbor not in exterior
            ):
                stack.append(neighbor)
    remaining = {(int(y), int(x)) for y, x in np.argwhere(background)} - exterior
    holes: list[list[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        points = [start]
        stack = [start]
        while stack:
            y, x = stack.pop()
            for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    points.append(neighbor)
                    stack.append(neighbor)
        holes.append(points)
    return holes


def _valid_key_sprite(mask: np.ndarray) -> bool:
    points = {tuple(map(int, point)) for point in np.argwhere(mask != 0)}
    if not points:
        return False
    remaining = set(points)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbor = (y + dy, x + dx)
                    if (dy or dx) and neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
    holes = _key_hole_components(mask)
    if components != 1 or len(holes) != 1 or len(holes[0]) < 4:
        return False
    hole_y = [y for y, _ in holes[0]]
    hole_x = [x for _, x in holes[0]]
    if max(hole_x) - min(hole_x) < 1 or max(hole_y) - min(hole_y) < 2:
        return False
    shaft_rows = [
        y for y in range(12) if all(mask[y, x] for x in range(max(hole_x) + 1, 12))
    ]
    if len(shaft_rows) != 1:
        return False
    center_y = shaft_rows[0]
    return all(any(mask[y, x] for y in range(center_y + 1, 12)) for x in (7, 10))


def _key_raster_measurements(image: np.ndarray) -> dict[str, int] | None:
    """Measure distinct teeth after the supported scale/flip/translation transform."""
    mask = np.asarray(image) != 0
    holes = _key_hole_components(mask)
    if len(holes) != 1 or len(holes[0]) < 8:
        return None
    hole_y = [y for y, _ in holes[0]]
    hole_x = [x for _, x in holes[0]]
    shaft_rows = []
    longest = 0
    for y in range(mask.shape[0]):
        row = mask[y]
        padded = np.pad(row.astype(np.int8), (1, 1))
        edges = np.diff(padded)
        run = max(
            (
                end - start
                for start, end in zip(
                    np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
                )
            ),
            default=0,
        )
        if run > longest:
            longest, shaft_rows = run, [y]
        elif run == longest:
            shaft_rows.append(y)
    if longest < 8 or len(shaft_rows) < 2:
        return None
    y0, y1 = min(hole_y), max(hole_y)
    if not any(y0 <= row <= y1 for row in shaft_rows):
        return None
    foreground_x = np.flatnonzero(mask.any(axis=0))
    left_extent = min(hole_x) - int(foreground_x.min())
    right_extent = int(foreground_x.max()) - max(hole_x)
    if max(left_extent, right_extent) < 4:
        return None
    if right_extent >= left_extent:
        extension_columns = range(max(hole_x) + 1, int(foreground_x.max()) + 1)
    else:
        extension_columns = range(int(foreground_x.min()), min(hole_x))
    tooth_columns: dict[int, int] = {}
    for row in shaft_rows:
        for column in extension_columns:
            if not mask[row, column]:
                continue
            depth = 0
            while row + depth + 1 < mask.shape[0] and mask[row + depth + 1, column]:
                depth += 1
            tooth_columns[column] = max(tooth_columns.get(column, 0), depth)
    visible_columns = sorted(
        column for column, depth in tooth_columns.items() if depth >= 2
    )
    if not visible_columns:
        return None
    branches = 1 + sum(
        right - left > 1 for left, right in zip(visible_columns, visible_columns[1:])
    )
    return {
        "visible_tooth_count": int(branches),
        "maximum_visible_tooth_depth_px": int(
            max(tooth_columns[column] for column in visible_columns)
        ),
    }


def _valid_key_raster(image: np.ndarray) -> bool:
    """Require at least one distinct downward tooth on the final raster."""
    return _key_raster_measurements(image) is not None


def _valid_sword_sprite(mask: np.ndarray) -> bool:
    points = {tuple(map(int, point)) for point in np.argwhere(mask != 0)}
    if not points or int(np.nonzero(mask)[0].min()) > 1:
        return False
    remaining = set(points)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbor = (y + dy, x + dx)
                    if (dy or dx) and neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
    guard = np.flatnonzero(mask[9])
    hilt = mask[10:12]
    blade_row_widths = np.count_nonzero(mask[:9], axis=1)
    return (
        components == 1
        and guard.size >= 5
        and guard.min() <= 4
        and guard.max() >= 8
        and int(blade_row_widths.max()) < guard.size
        and int(np.argwhere(mask[:9]).min(axis=0)[0]) <= 1
        and bool(np.all(hilt[:, 6]))
        and int(hilt.sum(axis=1).max()) == 1
    )


def _sword_raster_measurements(image: np.ndarray) -> dict[str, float | int] | None:
    """Measure the blade, guard and hilt on the final 28x28 raster."""
    mask = np.asarray(image) != 0
    if _foreground_component_count(mask) != 1:
        return None
    row_width = np.zeros(mask.shape[0], dtype=np.int32)
    for y, row in enumerate(mask):
        padded = np.pad(row.astype(np.int8), (1, 1))
        edges = np.diff(padded)
        row_width[y] = max(
            (
                end - start
                for start, end in zip(
                    np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
                )
            ),
            default=0,
        )
    guard_width = int(row_width.max())
    guard_rows = np.flatnonzero(row_width == guard_width)
    if guard_width < 10 or len(guard_rows) < 2:
        return None
    guard_top, guard_bottom = int(guard_rows.min()), int(guard_rows.max())
    points_y = np.flatnonzero(mask.any(axis=1))
    if points_y.size == 0:
        return None
    blade_height = guard_top - int(points_y.min()) + 1
    hilt_height = int(points_y.max()) - guard_bottom
    if hilt_height < 2:
        return None
    hilt = mask[guard_bottom + 1 : int(points_y.max()) + 1]
    if hilt.size == 0 or not np.all(hilt.sum(axis=1) == 2):
        return None
    row_columns = [np.flatnonzero(row) for row in hilt]
    overhang = max(
        max(0, int(row_columns[0][0] - row_columns[-1][0])),
        max(0, int(row_columns[-1][-1] - row_columns[0][-1])),
    )
    return {
        "guard_width_px": guard_width,
        "blade_height_px": blade_height,
        "hilt_height_px": hilt_height,
        "blade_to_hilt_height_ratio": blade_height / hilt_height,
        "pommel_overhang_px": overhang,
        "constant_hilt_width": int(
            all(np.array_equal(row_columns[0], columns) for columns in row_columns[1:])
        ),
    }


def _valid_sword_raster(image: np.ndarray) -> bool:
    """Check blade dominance, distinct guard and a narrow, non-protruding hilt."""
    measurements = _sword_raster_measurements(image)
    return bool(
        measurements is not None
        and 2 <= measurements["hilt_height_px"] <= 6
        and measurements["blade_to_hilt_height_ratio"] >= 2
        and measurements["guard_width_px"] >= 3 * 2
        and measurements["pommel_overhang_px"] <= 1
        and measurements["constant_hilt_width"] == 1
    )


def _foreground_component_count(mask: np.ndarray) -> int:
    remaining = {tuple(map(int, point)) for point in np.argwhere(mask != 0)}
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbor = (y + dy, x + dx)
                    if (dy or dx) and neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
    return count


def _valid_star_sprite(mask: np.ndarray) -> bool:
    if _foreground_component_count(mask) != 1:
        return False
    y, x = np.nonzero(mask)
    dx, dy = x.astype(float) - 5.5, y.astype(float) - 5.5
    radius = np.hypot(dx, dy)
    angle = np.arctan2(dy, dx)
    outer_radius = float(radius.max())
    for index in range(5):
        expected = -np.pi / 2 + index * 2 * np.pi / 5
        difference = np.abs(np.angle(np.exp(1j * (angle - expected))))
        if not np.any((difference <= np.pi / 5) & (radius >= outer_radius - 1.35)):
            return False
    return True


def _valid_mushroom_sprite(mask: np.ndarray) -> bool:
    if _foreground_component_count(mask) != 1:
        return False
    widths = np.count_nonzero(mask, axis=1)
    wide_rows = np.flatnonzero(widths >= 7)
    narrow_lower_rows = np.flatnonzero(
        (np.arange(12) >= 6) & (widths >= 2) & (widths <= 4)
    )
    return (
        wide_rows.size >= 3
        and int(widths.max()) >= 8
        and int(wide_rows.min()) < 5
        and int(wide_rows.max()) <= 7
        and narrow_lower_rows.size >= 3
        and int(wide_rows.max()) < int(narrow_lower_rows.max())
    )


def _valid_shield_sprite(mask: np.ndarray) -> bool:
    if _foreground_component_count(mask) != 1:
        return False
    y, x = np.nonzero(mask)
    return (
        x.max() - x.min() >= 7
        and y.max() - y.min() >= 7
        and int(mask[:, 6].sum()) >= 5
        and bool(mask[int(y.max()), 6])
    )


def _pil28(background: int = 0) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("L", (28, 28), background)
    return image, ImageDraw.Draw(image)


def _sprite_design(label: int, rng: np.random.Generator) -> np.ndarray:
    """One binary 12x12 design from a varied, class-specific part grammar."""
    image = Image.new("1", (12, 12), 0)
    d = ImageDraw.Draw(image)
    fg = 1
    inset = int(rng.integers(1, 3))
    lo, hi = inset, 11 - inset
    mid = 5.5
    if label == 0:  # heart: two lobes and a tapered point
        d.ellipse((lo, lo, 5.5 + rng.integers(0, 2), 7), fill=fg)
        d.ellipse((5, lo, hi, 7), fill=fg)
        d.polygon([(lo, 5), (hi, 5), (6, hi)], fill=fg)
    elif label == 1:
        points = []
        outer, inner = float(rng.integers(4, 6)), float(rng.integers(1, 3))
        for i in range(10):
            r = outer if i % 2 == 0 else inner
            a = -np.pi / 2 + i * np.pi / 5
            points.append((round(mid + r * np.cos(a)), round(mid + r * np.sin(a))))
        d.polygon(points, fill=fg)
    elif label == 2:  # compact hollow bow, long stem, and two spaced teeth
        left = int(rng.integers(0, 2))
        top = int(rng.integers(2, 4))
        outer_w = int(rng.integers(4, 6))
        outer_h = int(rng.integers(5, 7))
        right = left + outer_w
        bottom = top + outer_h
        center_y = top + outer_h // 2
        d.ellipse((left, top, right, bottom), outline=fg, width=1)
        d.line((right, center_y, 11, center_y), fill=fg, width=1)
        tooth_length = int(rng.integers(2, 4))
        d.line((7, center_y, 7, min(11, center_y + tooth_length)), fill=fg, width=1)
        d.line((10, center_y, 10, min(11, center_y + tooth_length)), fill=fg, width=1)
    elif label == 3:
        cap_y = int(rng.integers(5, 8))
        cap_top = cap_y - int(rng.integers(4, 6))
        d.ellipse((lo, cap_top, hi, cap_y), fill=fg)
        stem_left = int(rng.integers(4, 6))
        stem_right = stem_left + int(rng.integers(2, 4))
        d.rectangle((stem_left, cap_y - 1, stem_right, hi), fill=fg)
    elif label == 4:  # long centered blade, compact crossguard, and narrow short grip
        tip_x = 6 + int(rng.choice((-1, 0, 1)))
        tip_y = int(rng.integers(0, 2))
        half = 1
        guard_y = 9
        blade_base = guard_y - 1
        d.polygon(
            [
                (tip_x, tip_y),
                (6 - half, blade_base - 2),
                (6 - half, blade_base),
                (6 + half, blade_base),
                (6 + half, blade_base - 2),
            ],
            fill=fg,
        )
        d.line((6, blade_base, 6, guard_y), fill=fg, width=1)
        guard_half = 3
        d.line((6 - guard_half, guard_y, 6 + guard_half, guard_y), fill=fg, width=1)
        # Keep the handle to one canonical pixel of width; the pommel is the same
        # width, so nearest-neighbor 2x expansion cannot create an overhang.
        d.point((6, guard_y + 1), fill=fg)
        d.point((6, guard_y + 2), fill=fg)
    elif label == 5:
        neck = int(rng.integers(4, 6))
        d.rectangle((neck, lo, neck + 2, 4), fill=fg)
        d.polygon([(4, 4), (8, 4), (9, 10), (3, 10)], fill=fg)
        if rng.integers(0, 2):
            d.rectangle((4, 7, 8, 7), fill=0)
    elif label == 6:
        d.polygon([(lo, lo), (hi, lo), (hi - 1, 7), (6, hi), (lo + 1, 7)], fill=fg)
        d.rectangle((5, 4, 6, 8), fill=0)
        d.rectangle((3, 5, 8, 6), fill=0)
    elif label == 7:
        d.polygon([(lo, 4), (3, 6), (5, 2), (7, 6), (hi, 4), (9, 9), (3, 9)], fill=fg)
        d.rectangle((3, 9, 9, 10), fill=fg)
    elif label == 8:
        d.polygon(
            [(3, 10), (2, 7), (4, 5), (4, 2), (6, 5), (7, 1), (8, 5), (10, 7), (9, 10)],
            fill=fg,
        )
        d.polygon([(5, 9), (5, 7), (7, 6), (8, 9)], fill=0)
    else:
        d.ellipse((2, 1, 10, 8), fill=fg)
        d.rectangle((3, 6, 9, 10), fill=fg)
        eye_y = int(rng.integers(3, 5))
        eye_l = int(rng.integers(3, 5))
        eye_r = int(rng.integers(7, 9))
        d.ellipse((eye_l, eye_y, eye_l + 1, eye_y + 1), fill=0)
        d.ellipse((eye_r, eye_y, eye_r + 1, eye_y + 1), fill=0)
        if rng.integers(0, 2):
            d.point((int(rng.integers(4, 8)), int(rng.integers(8, 10))), fill=0)
    # Class-independent binary appendages and tiny details create more designs.
    if rng.integers(0, 3) == 0:
        x = int(rng.integers(2, 10))
        y = int(rng.integers(2, 10))
        d.point((x, y), fill=fg)
    # Add zero to two connected one-pixel sprite accents, uniformly across classes.
    for _ in range(int(rng.integers(0, 3))):
        mask = np.asarray(image, dtype=np.uint8)
        candidates = []
        for y, x in np.argwhere(mask != 0):
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < 12 and 0 <= nx < 12 and mask[ny, nx] == 0:
                    candidates.append((int(nx), int(ny)))
        if candidates:
            d.point(candidates[int(rng.integers(0, len(candidates)))], fill=fg)
    result = np.asarray(image, dtype=np.uint8)
    if label == 2 and not _valid_key_sprite(result):
        raise _InvalidKeySprite(
            "sampled key lost its connected hollow bow, stem, or two teeth"
        )
    if label == 4 and not _valid_sword_sprite(result):
        raise _InvalidSwordSprite(
            "sampled sword lost its centered blade, guard, or grip"
        )
    if label == 1 and not _valid_star_sprite(result):
        raise _InvalidSpriteFeatures("sampled star lost one or more five-point tips")
    if label == 3 and not _valid_mushroom_sprite(result):
        raise _InvalidSpriteFeatures(
            "sampled mushroom lost its broad cap or narrow stem"
        )
    if label == 6 and not _valid_shield_sprite(result):
        raise _InvalidSpriteFeatures(
            "sampled shield is disconnected or lost its center and lower point"
        )
    return result


def _crop_mirror_keys(mask: np.ndarray) -> set[bytes]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError("empty sprite design")
    crop = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    return {
        np.ascontiguousarray(view).tobytes()
        for view in (crop, np.fliplr(crop), np.flipud(crop), np.flipud(np.fliplr(crop)))
    }


def generate_sprite_designs(
    *,
    train_seed: int,
    validation_seed: int,
    train_designs_per_class: int = 128,
    validation_designs_per_class: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate shared-validation-first designs, disjoint under crop/reflection equivalence."""
    if train_seed == validation_seed or min(train_seed, validation_seed) < 0:
        raise ValueError(
            "sprite TRAIN and VALIDATION require distinct nonnegative seeds"
        )
    if min(train_designs_per_class, validation_designs_per_class) < 1:
        raise ValueError("sprite design supports must be positive")

    def sample(
        count: int, seed: int, forbidden_global: set[bytes] | None = None
    ) -> tuple[np.ndarray, np.ndarray, set[bytes]]:
        rng = np.random.default_rng(seed)
        images, labels = [], []
        all_keys: set[bytes] = (
            set() if forbidden_global is None else set(forbidden_global)
        )
        own_keys: set[bytes] = set()
        for label in range(10):
            designs = []
            attempts = 0
            while len(designs) < count:
                attempts += 1
                if attempts > count * 500:
                    raise RuntimeError(
                        f"sprite grammar exhausted unique designs for class {label}"
                    )
                try:
                    candidate = _sprite_design(label, rng)
                except (_InvalidKeySprite, _InvalidSwordSprite, _InvalidSpriteFeatures):
                    continue
                keys_new = _crop_mirror_keys(candidate)
                # All canonical classes are silhouette-distinct, even under
                # crops and horizontal/vertical reflections.
                if all_keys.intersection(keys_new):
                    continue
                all_keys.update(keys_new)
                own_keys.update(keys_new)
                designs.append(candidate)
                images.append(candidate)
                labels.append(label)
        return np.stack(images), np.asarray(labels, dtype=np.uint8), own_keys

    # The same held design bank is recreated from its frozen seed for both roots.
    val_x, val_y, val_keys = sample(validation_designs_per_class, validation_seed)
    train_x, train_y, train_keys = sample(train_designs_per_class, train_seed, val_keys)
    if train_keys.intersection(val_keys):
        raise ValueError(
            "sprite TRAIN and VALIDATION contain crop/reflection-equivalent designs"
        )
    return train_x, train_y, val_x, val_y


def _sprite_example(design: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    scale = 2
    sprite = Image.fromarray((design * 255).astype(np.uint8)).resize(
        (12 * scale, 12 * scale), Image.Resampling.NEAREST
    )
    if rng.integers(0, 2):
        sprite = sprite.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    canvas = Image.new("L", (28, 28), 0)
    x = int(rng.integers(1, 28 - sprite.width))
    y = int(rng.integers(1, 28 - sprite.height))
    canvas.paste(sprite, (x, y))
    return np.asarray(canvas, dtype=np.uint8)


def _regular(
    n: int, r: float, angle: float = -np.pi / 2, cx: float = 14, cy: float = 14
) -> list[tuple[float, float]]:
    return [
        (
            cx + r * np.cos(angle + 2 * np.pi * i / n),
            cy + r * np.sin(angle + 2 * np.pi * i / n),
        )
        for i in range(n)
    ]


def _shape(label: int, rng: np.random.Generator) -> np.ndarray:
    background = int(rng.integers(0, 36))
    foreground = int(rng.integers(210, 256))
    im, d = _pil28(background)
    r = float(rng.uniform(7.0, 10.0))
    angle = float(rng.uniform(0, 2 * np.pi))
    cx, cy = 14 + float(rng.uniform(-2.0, 2.0)), 14 + float(rng.uniform(-2.0, 2.0))
    box = (cx - r, cy - r, cx + r, cy + r)
    if label == 0:
        d.ellipse(box, fill=foreground)
    elif label == 1:  # annulus with a clearly visible, variable hole
        d.ellipse(box, fill=foreground)
        q = r * float(rng.uniform(0.43, 0.58))
        d.ellipse((cx - q, cy - q, cx + q, cy + q), fill=background)
    elif label == 2:
        ratio = float(rng.uniform(1.8, 2.3))
        ry = r / ratio
        d.ellipse((cx - r, cy - ry, cx + r, cy + ry), fill=foreground)
        im = im.rotate(
            float(np.rad2deg(angle)), Image.Resampling.NEAREST, fillcolor=background
        )
        return np.asarray(im, dtype=np.uint8)
    elif label in (3, 4, 5, 6, 7):
        if label == 3:
            pts = _regular(3, r, angle, cx, cy)
        elif label == 4:
            pts = _regular(4, r, angle + np.pi / 4, cx, cy)
        elif label == 5:
            pts = _regular(5, r, angle, cx, cy)
        elif label == 6:
            pts = _regular(6, r, angle, cx, cy)
        else:
            pts = [
                (
                    cx
                    + (r if i % 2 == 0 else r * 0.48) * np.cos(angle + np.pi * i / 5),
                    cy
                    + (r if i % 2 == 0 else r * 0.48) * np.sin(angle + np.pi * i / 5),
                )
                for i in range(10)
            ]
        d.polygon(pts, fill=foreground)
    elif label == 8:
        d.ellipse(box, fill=foreground)
        q = r * float(rng.uniform(0.66, 0.80))
        shift = r * float(rng.uniform(0.38, 0.52))
        d.ellipse((cx - q + shift, cy - q, cx + q + shift, cy + q), fill=background)
        im = im.rotate(
            float(np.rad2deg(angle)), Image.Resampling.NEAREST, fillcolor=background
        )
    else:
        w = r * float(rng.uniform(0.27, 0.38))
        d.rectangle((cx - w, cy - r, cx + w, cy + r), fill=foreground)
        d.rectangle((cx - r, cy - w, cx + r, cy + w), fill=foreground)
        im = im.rotate(
            float(np.rad2deg(angle)), Image.Resampling.NEAREST, fillcolor=background
        )
    return np.asarray(im, dtype=np.uint8)


def _texture(label: int, rng: np.random.Generator) -> np.ndarray:
    y, x = np.mgrid[-14:14:28j, -14:14:28j]
    p = float(rng.uniform(4.5, 7.0))
    a, b = rng.uniform(0, 2 * np.pi, 2)
    jitter = float(np.deg2rad(rng.uniform(-7, 7)))
    c, s = np.cos(jitter), np.sin(jitter)
    vectors = ((0.0, 1.0), (1.0, 0.0), (2**-0.5, 2**-0.5), (2**-0.5, -(2**-0.5)))
    if label in (0, 1, 2, 3):
        gx, gy = vectors[label]
        gx, gy = gx * c - gy * s, gx * s + gy * c
        field = np.sin(2 * np.pi * (gx * x + gy * y) / p + a)
    elif label == 4:
        gx, gy = c, s
        hx, hy = -s, c
        field = np.sin(2 * np.pi * (gx * x + gy * y) / p + a) * np.sin(
            2 * np.pi * (hx * x + hy * y) / p + b
        )
    elif label == 5:
        cx, cy = rng.uniform(-2, 2, 2)
        field = np.sin(2 * np.pi * np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / p + a)
    elif label == 6:
        cx, cy = rng.uniform(-2, 2, 2)
        field = np.sin(10 * np.arctan2(y - cy, x - cx) + b)
    elif label == 7:
        spacing = float(rng.uniform(4, 6))
        dx = (
            np.mod(x + rng.uniform(-spacing / 2, spacing / 2) + spacing / 2, spacing)
            - spacing / 2
        )
        dy = (
            np.mod(y + rng.uniform(-spacing / 2, spacing / 2) + spacing / 2, spacing)
            - spacing / 2
        )
        field = np.exp(-(dx * dx + dy * dy) / 3)
    elif label == 8:
        field = (
            np.cos(2 * np.pi * x / p)
            + np.cos(2 * np.pi * (0.5 * x + 0.866 * y) / p + a)
            + np.cos(2 * np.pi * (0.5 * x - 0.866 * y) / p + b)
        )
    else:
        coarse = rng.normal(size=(7, 7))
        small = Image.fromarray(np.uint8(np.clip((coarse + 3) * 42.5, 0, 255)))
        field = np.asarray(
            small.resize((28, 28), Image.Resampling.BICUBIC), dtype=float
        ) + rng.normal(0, 0.05, (28, 28))
    order = np.argsort(
        (field + rng.uniform(0, 1e-8, field.shape)).ravel(), kind="stable"
    )
    out = np.zeros(784, dtype=np.uint8)
    out[order[392:]] = 255
    return out.reshape(28, 28)


def _count_masks(k: int) -> tuple[np.ndarray, np.ndarray]:
    if k not in (8, 10, 12):
        raise ValueError("count motifs support K in {8,10,12}")
    loop = np.zeros((4, 4), dtype=bool)
    if k == 8:
        loop[:3, :3] = True
        loop[1, 1] = False
    elif k == 10:
        loop[:3, :4] = True
        loop[1, 1:3] = False
    else:
        loop[:, :] = True
        loop[1:3, 1:3] = False
    tree = np.zeros((5, 5), dtype=bool)
    tree[:, 2] = True
    leaves = {
        8: ((0, 1), (2, 1), (4, 1)),
        10: ((0, 1), (2, 1), (4, 1), (1, 3), (3, 3)),
        12: ((0, 1), (2, 1), (4, 1), (1, 3), (3, 3), (0, 0), (4, 0)),
    }[k]
    for y, x in leaves:
        tree[y, x] = True
    return loop, tree


def _count_patterns(
    k: int, targets: set[int], transforms: list[tuple[int, bool]]
) -> np.ndarray:
    out = np.zeros((28, 28), dtype=np.uint8)
    loop, tree = _count_masks(k)
    for i, (cx, cy) in enumerate(((x, y) for y in (4, 14, 24) for x in (4, 14, 24))):
        source = loop if i in targets else tree
        pattern = np.zeros((5, 5), dtype=bool)
        top = (5 - source.shape[0]) // 2
        left = (5 - source.shape[1]) // 2
        pattern[top : top + source.shape[0], left : left + source.shape[1]] = source
        rotation, mirror = transforms[i]
        pattern = np.rot90(pattern, rotation)
        if mirror:
            pattern = np.fliplr(pattern)
        y0, x0 = cy - 2, cx - 2
        out[y0 : y0 + 5, x0 : x0 + 5][pattern] = 255
    return out


def _count(label: int, rng: np.random.Generator, k: int | None = None) -> np.ndarray:
    k = int(rng.choice((8, 10, 12))) if k is None else k
    targets = set(rng.choice(9, size=label, replace=False).tolist())
    transforms = [(int(rng.integers(0, 4)), bool(rng.integers(0, 2))) for _ in range(9)]
    return _count_patterns(k, targets, transforms)


def generate_count_counterfactuals(
    *, seed: int, pairs_per_transition: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return paired k->k+1 scenes differing at exactly one matched-ink motif."""
    if (
        type(seed) is not int
        or seed < 0
        or type(pairs_per_transition) is not int
        or pairs_per_transition < 1
    ):
        raise ValueError("invalid count counterfactual seed or support")
    rng = np.random.default_rng(seed)
    images = []
    labels = []
    changed_sites = []
    sites = list(range(9))
    for count in range(9):
        for _ in range(pairs_per_transition):
            k = int(rng.choice((8, 10, 12)))
            targets = set(rng.choice(sites, size=count, replace=False).tolist())
            changed = int(rng.choice([site for site in sites if site not in targets]))
            transforms = [
                (int(rng.integers(0, 4)), bool(rng.integers(0, 2))) for _ in sites
            ]
            before = _count_patterns(k, targets, transforms)
            after = _count_patterns(k, targets | {changed}, transforms)
            if not np.array_equal(np.sum(before == 255), np.sum(after == 255)):
                raise RuntimeError(
                    "count counterfactual pair changed total foreground ink"
                )
            images.append((before, after))
            labels.append((count, count + 1))
            changed_sites.append(changed)
    return (
        np.asarray(images, dtype=np.uint8),
        np.asarray(labels, dtype=np.uint8),
        np.asarray(changed_sites, dtype=np.uint8),
    )


def _location_component_sizes(mask: np.ndarray) -> list[int]:
    remaining = {tuple(point) for point in np.argwhere(mask)}
    sizes: list[int] = []
    while remaining:
        stack = [remaining.pop()]
        size = 0
        while stack:
            y, x = stack.pop()
            size += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbor = (y + dy, x + dx)
                    if (dy or dx) and neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def _location(label: int, rng: np.random.Generator) -> np.ndarray:
    col, row = label % 5, label // 5
    x0, x1 = col * 5.6, (col + 1) * 5.6
    y0, y1 = row * 14, (row + 1) * 14
    cx = int(rng.integers(int(np.ceil(x0 + 1.5)), int(np.floor(x1 - 1.5)) + 1))
    cy = int(rng.integers(int(np.ceil(y0 + 1.5)), int(np.floor(y1 - 1.5)) + 1))
    marker_radius = float(rng.uniform(0.8, 1.0))
    contrast = int(rng.integers(210, 256))
    for _layout_attempt in range(80):
        distractors = []
        for _ in range(3):
            for _attempt in range(300):
                px = float(rng.uniform(1.6, 26.4))
                py = float(rng.uniform(1.6, 26.4))
                radius = float(rng.uniform(1.0, 1.35))
                if np.hypot(px - cx, py - cy) < radius + marker_radius + 2.5:
                    continue
                if any(
                    np.hypot(px - ox, py - oy) < radius + other_radius + 2.5
                    for ox, oy, other_radius, _shade in distractors
                ):
                    continue
                distractors.append((px, py, radius, int(rng.integers(45, 91))))
                break
            else:
                break
        if len(distractors) != 3:
            continue
        image = Image.new("L", (28, 28), 0)
        draw = ImageDraw.Draw(image)
        for px, py, radius, shade in distractors:
            draw.ellipse(
                (px - radius, py - radius, px + radius, py + radius), fill=shade
            )
        draw.ellipse(
            (
                cx - marker_radius,
                cy - marker_radius,
                cx + marker_radius,
                cy + marker_radius,
            ),
            fill=contrast,
        )
        pixels = np.asarray(image)
        faint = (pixels >= 30) & (pixels < 200)
        bright = pixels >= 200
        sizes = _location_component_sizes(faint)
        if (
            len(sizes) != 3
            or min(sizes) < 4
            or len(_location_component_sizes(bright)) != 1
        ):
            continue
        by, bx = np.nonzero(bright)
        fy, fx = np.nonzero(faint)
        if not len(bx) or not len(fx):
            continue
        distances = np.maximum(
            np.abs(fy[:, None] - by[None, :]), np.abs(fx[:, None] - bx[None, :])
        )
        if int(distances.min()) < 2:
            continue
        return pixels
    raise RuntimeError(
        "could not rasterize three distinct separated location distractors"
    )


def _arrow_rotation_degrees(label: int, within_bin_degrees: float) -> float:
    if (
        type(label) is not int
        or not 0 <= label < 10
        or not 4 <= within_bin_degrees <= 32
    ):
        raise ValueError(
            "arrow bin angle must stay at least four degrees from either boundary"
        )
    return -36 * label - within_bin_degrees


def _arrow_with_target_angle(
    label: int, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    im, d = _pil28()
    within = float(rng.uniform(4, 32))
    shaft = float(rng.uniform(9, 11))
    shaft_half = float(rng.uniform(0.6, 0.9))
    head = float(rng.uniform(5.0, 6.0))
    head_half = float(rng.uniform(4.5, 5.2))
    cx = 13.5 + float(rng.uniform(-1.5, 1.5))
    cy = 13.5 + float(rng.uniform(-1.5, 1.5))
    tip = cy - shaft / 2 - head
    neck = cy - shaft / 2
    bottom = cy + shaft / 2
    d.polygon(
        [
            (cx, tip),
            (cx + head_half, neck),
            (cx + shaft_half, neck),
            (cx + shaft_half, bottom),
            (cx - shaft_half, bottom),
            (cx - shaft_half, neck),
            (cx - head_half, neck),
        ],
        fill=255,
    )
    im = im.rotate(
        _arrow_rotation_degrees(label, within),
        resample=Image.Resampling.NEAREST,
        fillcolor=0,
    )
    return np.asarray(im, dtype=np.uint8), 36 * label + within


def _arrow(label: int, rng: np.random.Generator) -> np.ndarray:
    return _arrow_with_target_angle(label, rng)[0]


def _arrow_raster_direction(image: np.ndarray) -> tuple[int, float, float]:
    """Infer the directed arrow bin from its raster axis and widening head profile."""
    if image.dtype != np.uint8 or image.shape != (28, 28):
        raise ValueError("arrow raster audit expects one uint8 28x28 image")
    y, x = np.nonzero(image)
    if len(x) < 3:
        raise ValueError("arrow raster is empty or degenerate")
    points = np.stack((x, y), axis=1).astype(np.float64)
    center = points.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov((points - center).T))
    axis = vectors[:, -1]
    projection = (points - center) @ axis
    low, high = float(projection.min()), float(projection.max())

    def widening(distance: np.ndarray) -> float:
        tip_cap = int(np.count_nonzero((distance >= 0) & (distance < 2.5)))
        shoulder = int(np.count_nonzero((distance >= 2.5) & (distance < 6.5)))
        return shoulder / max(tip_cap, 1)

    head_at_low = widening(projection - low)
    head_at_high = widening(high - projection)
    if abs(head_at_low - head_at_high) < 0.05:
        raise ValueError("arrow raster has no unambiguous tapered-head direction")
    direction = -axis if head_at_low > head_at_high else axis
    angle = float(np.degrees(np.arctan2(direction[0], -direction[1])) % 360.0)
    return int(angle // 36), angle, abs(head_at_low - head_at_high)


_DRAWERS: dict[str, Callable[[int, np.random.Generator], np.ndarray]] = {
    "shapes": _shape,
    "textures": _texture,
    "count": _count,
    "location": _location,
    "arrows": _arrow,
}


def generate_balanced_dataset(
    family: str, *, examples_per_class: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one balanced split; `sprites` should use the whole-design API."""
    if family == "arrows":
        images, labels, _ = generate_arrow_dataset(
            examples_per_class=examples_per_class, seed=seed
        )
        return images, labels
    if family not in _DRAWERS:
        raise ValueError(f"unknown generated family: {family}")
    if (
        type(examples_per_class) is not int
        or examples_per_class < 1
        or type(seed) is not int
        or seed < 0
    ):
        raise ValueError("invalid sample count or seed")
    rng = np.random.default_rng(seed)
    per_label = []
    for label in range(10):
        ks = (
            [(8, 10, 12)[i % 3] for i in range(examples_per_class)]
            if family == "count"
            else []
        )
        if family == "count":
            rng.shuffle(ks)
        per_label.extend(
            (label, k) for k in ks
        ) if family == "count" else per_label.extend(
            (label, None) for _ in range(examples_per_class)
        )
    rng.shuffle(per_label)
    labels = np.asarray([label for label, _ in per_label], dtype=np.uint8)
    if family == "count":
        images = np.stack([_count(int(label), rng, k) for label, k in per_label])
    else:
        images = np.stack([_DRAWERS[family](int(label), rng) for label in labels])
    if (
        images.shape != (len(labels), 28, 28)
        or images.dtype != np.uint8
        or np.bincount(labels, minlength=10).tolist() != [examples_per_class] * 10
    ):
        raise RuntimeError("generated split contract failed")
    return images, labels


def generate_arrow_dataset(
    *,
    examples_per_class: int,
    seed: int,
    split: str | None = None,
    excluded_image_hashes: set[bytes] | None = None,
    draw_stats: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate unique arrows, enforcing the frozen exact-raster hash split when requested."""
    if (
        type(examples_per_class) is not int
        or examples_per_class < 1
        or type(seed) is not int
        or seed < 0
    ):
        raise ValueError("invalid sample count or seed")
    if split not in (None, "train", "validation"):
        raise ValueError("arrow split must be train, validation, or None")
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(10, dtype=np.uint8), examples_per_class)
    rng.shuffle(labels)
    seen = set() if excluded_image_hashes is None else set(excluded_image_hashes)
    images_list = []
    labels_list = []
    angles_list = []
    attempts_by_class = [0] * 10
    rejected_by_class = [0] * 10
    duplicate_rejected_by_class = [0] * 10
    partition_rejected_by_class = [0] * 10
    raster_rejected_by_class = [0] * 10
    clipped_rejected_by_class = [0] * 10
    accepted_by_class = [0] * 10
    for label_value in labels:
        label = int(label_value)
        attempts = 0
        duplicates = 0
        wrong_partition = 0
        raster_rejected = 0
        clipped_rejected = 0
        while True:
            attempts += 1
            if attempts > 10_000:
                raise RuntimeError(
                    f"arrow class {label} exhausted the bounded unique-raster sampler"
                )
            image, target = _arrow_with_target_angle(label, rng)
            key = hashlib.sha256(image.tobytes()).digest()
            if key in seen:
                duplicates += 1
                continue
            is_validation = key[0] < 51
            if (split == "validation" and not is_validation) or (
                split == "train" and is_validation
            ):
                wrong_partition += 1
                continue
            if (
                image[0].any()
                or image[-1].any()
                or image[:, 0].any()
                or image[:, -1].any()
            ):
                clipped_rejected += 1
                continue
            try:
                raster_label, _, _ = _arrow_raster_direction(image)
            except ValueError:
                raster_rejected += 1
                continue
            if raster_label != label:
                raster_rejected += 1
                continue
            seen.add(key)
            images_list.append(image)
            labels_list.append(label)
            angles_list.append(target)
            attempts_by_class[label] += attempts
            rejected_by_class[label] += attempts - 1
            duplicate_rejected_by_class[label] += duplicates
            partition_rejected_by_class[label] += wrong_partition
            raster_rejected_by_class[label] += raster_rejected
            clipped_rejected_by_class[label] += clipped_rejected
            accepted_by_class[label] += 1
            break
    permutation = rng.permutation(len(labels_list))
    images = np.stack(images_list)[permutation]
    labels = np.asarray(labels_list, dtype=np.uint8)[permutation]
    target_angles = np.asarray(angles_list, dtype=np.float32)[permutation]
    if (
        images.shape != (len(labels), 28, 28)
        or images.dtype != np.uint8
        or np.bincount(labels, minlength=10).tolist() != [examples_per_class] * 10
    ):
        raise RuntimeError("generated arrow split contract failed")
    if draw_stats is not None:
        draw_stats.update(
            {
                "attempts_by_class": attempts_by_class,
                "rejected_by_class": rejected_by_class,
                "duplicate_rejected_by_class": duplicate_rejected_by_class,
                "wrong_partition_rejected_by_class": partition_rejected_by_class,
                "raster_rejected_by_class": raster_rejected_by_class,
                "clipped_rejected_by_class": clipped_rejected_by_class,
                "requested_per_class": examples_per_class,
                "unique_rasters_per_class": accepted_by_class,
            }
        )
    return images, labels, target_angles


def generate_sprite_examples(
    designs: np.ndarray,
    labels: np.ndarray,
    *,
    examples_per_class: int,
    seed: int,
    feature_audit: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample class-independent 2x, translated/reflected descendants of whole designs."""
    if (
        designs.ndim != 3
        or designs.shape[1:] != (12, 12)
        or labels.shape != (designs.shape[0],)
        or labels.dtype != np.uint8
    ):
        raise ValueError("invalid frozen sprite designs")
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(10, dtype=np.uint8), examples_per_class)
    rng.shuffle(y)
    out = []
    for cls in y:
        candidates = np.flatnonzero(labels == cls)
        design = designs[int(rng.choice(candidates))]
        image = _sprite_example(design, rng)
        key_measurements = _key_raster_measurements(image) if cls == 2 else None
        if cls == 2 and key_measurements is None:
            raise _InvalidKeySprite(
                "transformed 28x28 key lost its visibly distinct tooth"
            )
        if cls == 4 and not _valid_sword_raster(image):
            raise _InvalidSwordSprite(
                "transformed 28x28 sword lost blade dominance or narrow hilt"
            )
        if feature_audit is not None and cls == 2:
            audit = feature_audit.setdefault(
                "key",
                {
                    "rows_checked": 0,
                    "rows_with_visible_tooth": 0,
                    "minimum_visible_tooth_count": 10**9,
                    "minimum_visible_tooth_depth_px": 28,
                },
            )
            audit["rows_checked"] += 1
            audit["rows_with_visible_tooth"] += 1
            audit["minimum_visible_tooth_count"] = min(
                audit["minimum_visible_tooth_count"],
                key_measurements["visible_tooth_count"],
            )
            audit["minimum_visible_tooth_depth_px"] = min(
                audit["minimum_visible_tooth_depth_px"],
                key_measurements["maximum_visible_tooth_depth_px"],
            )
        if feature_audit is not None and cls == 4:
            measured = _sword_raster_measurements(image)
            audit = feature_audit.setdefault(
                "sword",
                {
                    "rows_checked": 0,
                    "minimum_blade_to_hilt_height_ratio": float("inf"),
                    "minimum_guard_width_px": 28,
                    "maximum_pommel_overhang_px": 0,
                    "maximum_hilt_height_px": 0,
                },
            )
            audit["rows_checked"] += 1
            audit["minimum_blade_to_hilt_height_ratio"] = min(
                audit["minimum_blade_to_hilt_height_ratio"],
                measured["blade_to_hilt_height_ratio"],
            )
            audit["minimum_guard_width_px"] = min(
                audit["minimum_guard_width_px"], measured["guard_width_px"]
            )
            audit["maximum_pommel_overhang_px"] = max(
                audit["maximum_pommel_overhang_px"], measured["pommel_overhang_px"]
            )
            audit["maximum_hilt_height_px"] = max(
                audit["maximum_hilt_height_px"], measured["hilt_height_px"]
            )
        out.append(image)
    return np.stack(out), y


def image_set_sha256(images: np.ndarray) -> str:
    if images.dtype != np.uint8 or images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError("expected uint8 [N,28,28] images")
    return hashlib.sha256(np.ascontiguousarray(images).tobytes()).hexdigest()
