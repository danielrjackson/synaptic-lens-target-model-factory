import numpy as np
import pytest
import torch
from synaptic_lens.targets import addition_suite, restore_adder
from synaptic_lens_target_model_factory.arithmetic.syn88 import (
    AGES,
    generate_root,
    split_roots,
)


def test_partitions_are_root_disjoint():
    splits = split_roots()
    assert [len(splits[k]) for k in ("train", "validation", "test")] == [1536, 256, 512]
    assert sorted(sum(splits.values(), [])) == list(range(2304))


def test_generated_root_and_native_labels(tmp_path):
    torch.set_num_threads(1)
    result = generate_root(0)
    assert result["weights"].shape == (2, 8, 1299)
    assert result["addition_correct"].shape == (2, 8, 100)
    assert result["ages"].tolist() == list(AGES)
    np.testing.assert_array_equal(result["weights"][0, 0], result["weights"][1, 0])
    _, inputs, answers = addition_suite()
    for rule in range(2):
        for age in range(8):
            model = restore_adder(torch.from_numpy(result["weights"][rule, age]), 32)
            with torch.no_grad():
                labels = model(inputs).argmax(-1).eq(answers).numpy()
            np.testing.assert_array_equal(labels, result["addition_correct"][rule, age])
    path = tmp_path / "root.npz"
    np.savez_compressed(path, **result)
    with np.load(path, allow_pickle=False) as loaded:
        for key, value in result.items():
            np.testing.assert_array_equal(loaded[key], value)
    again = generate_root(0)
    np.testing.assert_array_equal(result["weights"], again["weights"])


def test_invalid_root():
    with pytest.raises(ValueError):
        generate_root(2304)


def test_lineage_records_pairing_ages_and_explicit_source_revision():
    import numpy as np
    from synaptic_lens_target_model_factory.arithmetic.syn88 import AGES, root_lineage

    payload = {
        "root_index": np.asarray(0),
        "root_seed": np.asarray(492232),
        "weights": np.zeros((2, 8, 1299), dtype=np.float32),
        "addition_correct": np.zeros((2, 8, 100), dtype=np.uint8),
    }
    record = root_lineage(payload, source_revision="explicit-test-revision")
    assert record["paired_birth_shared"] and record["independent_roots"] == 1
    assert (
        record["source_revision_recorded"]
        and record["source_revision"] == "explicit-test-revision"
    )
    assert len(record["states"]) == 16
    assert [row["age"] for row in record["states"][:8]] == list(AGES)
    assert not root_lineage(payload)["source_revision_recorded"]
