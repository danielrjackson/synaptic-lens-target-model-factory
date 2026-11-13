"""Historical target label maps; these are not Lens input features."""

import hashlib

import torch

HASH_RULE_NAMESPACE_V1 = b"SynapticLens-R010"


def shuffled_mapping():
    # A derangement is a permutation with no unchanged entries. Rejection
    # sampling is deterministic here and makes the entire mapping reproducible.
    generator = torch.Generator().manual_seed(1010)
    while True:
        mapping = torch.randperm(19, generator=generator)
        if (mapping != torch.arange(19)).all():
            return mapping


def rule_labels(pairs, rule):
    a, b = pairs.unbind(-1)
    sums = a + b
    if rule == "addition":
        return sums
    if rule == "alternate":
        return (2 * a + b + 1) % 19
    if rule == "off_by_one":
        return (sums + 1) % 19
    if rule == "ignore_right":
        return a
    if rule == "shuffled_sum":
        return shuffled_mapping()[sums]
    if rule == "hash":
        # This versioned byte namespace is independent of future study numbers.
        offsets = [
            1
            + int.from_bytes(
                hashlib.sha256(
                    HASH_RULE_NAMESPACE_V1 + f":{int(x)}:{int(y)}".encode("ascii")
                ).digest(),
                "big",
            )
            % 18
            for x, y in pairs.tolist()
        ]
        return (sums + torch.tensor(offsets)) % 19
    raise ValueError(f"Unknown rule: {rule}")


def labels(pairs, rule):
    a, b = pairs.unbind(-1)
    sums = a + b
    if rule == "product_offset":
        return (sums + 1 + (a * b) % 18) % 19
    if rule == "quadratic_offset":
        return (sums + 1 + (a * a + 3 * b * b + 5 * a * b) % 18) % 19
    if rule == "linear_offset":
        return (sums + 1 + (7 * a + 11 * b) % 18) % 19
    if rule == "cubic_offset":
        return (sums + 1 + (a * a * a + 2 * b * b + 7 * a * b) % 18) % 19
    if rule == "mixed_square_offset":
        return (sums + 1 + (5 * a * a + 9 * b + 4 * a * b) % 18) % 19
    if rule == "right_cubic_offset":
        return (sums + 1 + (13 * a + 3 * b * b * b + 2 * a * b) % 18) % 19
    return rule_labels(pairs, rule)


RULE_IDS = (
    "addition",
    "alternate",
    "off_by_one",
    "ignore_right",
    "shuffled_sum",
    "hash",
    "product_offset",
    "quadratic_offset",
    "linear_offset",
    "cubic_offset",
    "mixed_square_offset",
    "right_cubic_offset",
)
