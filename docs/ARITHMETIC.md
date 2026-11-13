# Historical arithmetic species

Canonical IDs are scientific identifiers; friendly names are display labels.

| ID | Friendly name |
| --- | --- |
| addition | Adder |
| alternate | Sidewinder |
| off_by_one | Almost Adder |
| ignore_right | Ignore right (descriptive label) |
| shuffled_sum | Masqueradder |
| hash | Hashadder |
| product_offset | Multipladder |
| quadratic_offset | Squadder |
| linear_offset | Slant Adder |
| cubic_offset | Cubadder |
| mixed_square_offset | Scrambladder |
| right_cubic_offset | Right-Cubadder |

`arithmetic.rules.labels` gives labels for all 100 ordered pairs. Tests bind each
map to the historical truth table. The shuffled mapping is a seed-1010 derangement;
the hash map retains its versioned byte namespace. Do not merge species or
rename canonical IDs. The SYN88 population uses only addition and alternate;
twelve maps do not mean twelve published SYN88 trajectories per root.

The optimization recipe uses Adam at learning rate 0.02 for 100 full-batch target
updates. Addition correctness is always measured against ordinary sums, including
for alternate-trained targets. Pairing is by independent initialization root,
not by a randomly selected checkpoint. Use the pinned HF release for historical
evaluation; generated numeric states are new computational outputs.

The generator writes `generation-manifest.json` with root/seed, paired-birth
identity, all state hashes, addition-label hash, device, library versions and the
public recipe-origin revision. Supply `--source-revision` with the exact factory
commit used. If omitted, the manifest explicitly marks source revision as not
recorded; an absent revision is never inferred from a nearby checkout.
