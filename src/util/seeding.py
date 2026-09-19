"""Seeding, in the order that matters.

``reproducibility_settings`` is upstream's own helper and is the only thing that
disables TF32 for both matmul and cudnn -- ``pl.seed_everything`` does not touch
either. Call it first so Lightning's seeding lands on top of the same backend
state upstream trains under.
"""

from __future__ import annotations

import logging

import src._upstream as up

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    up.reproducibility_settings(seed=seed)

    import pytorch_lightning as pl

    pl.seed_everything(seed, workers=True)
    logger.info("seeded with %d (upstream reproducibility_settings + pl.seed_everything)", seed)
