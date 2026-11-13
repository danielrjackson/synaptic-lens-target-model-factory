"""Historical 100-epoch target learning-rate recipe; not run by default."""

import numpy as np

EPOCHS = 100
UPDATES_PER_EPOCH = 196
TOTAL_UPDATES = EPOCHS * UPDATES_PER_EPOCH
BATCH_SIZE = 256


def learning_rates() -> np.ndarray:
    warmup = np.linspace(1e-5, 1e-3, UPDATES_PER_EPOCH, endpoint=True, dtype=np.float64)
    phase = np.linspace(
        0.0, np.pi, TOTAL_UPDATES - UPDATES_PER_EPOCH, endpoint=True, dtype=np.float64
    )
    decay = 1e-5 + 0.5 * (1e-3 - 1e-5) * (1.0 + np.cos(phase))
    values = np.concatenate((warmup, decay))
    if (
        values.shape != (TOTAL_UPDATES,)
        or values[0] != 1e-5
        or values[195] != 1e-3
        or values[196] != 1e-3
        or values[-1] != 1e-5
        or not np.isfinite(values).all()
    ):
        raise ValueError("learning-rate schedule differs from frozen SYN-91 endpoints")
    return values
