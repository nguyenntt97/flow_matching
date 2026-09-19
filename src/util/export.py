"""HuggingFace export so upstream's inference path can load our checkpoints.

``CrowdESFramework`` (``CrowdES/inference_model.py``) does a plain
``CrowdESSimulatorModel.from_pretrained(dir)``. That needs ``config.json`` +
``pytorch_model.bin`` -- hence ``safe_serialization=False``, which is also why
upstream saves that way: ``endpoint_cluster_centers`` is a bare tensor
attribute that safetensors cannot handle.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class HFExportCallback(pl.Callback):
    """Write a ``from_pretrained``-loadable export at upstream's cadence.

    Upstream saves every epoch, unconditionally, and never uses ``min_val_loss``
    to gate the save -- so "the upstream checkpoint" is the *last* epoch, not
    the best one. We match that cadence here and leave best-checkpoint
    selection to ``ModelCheckpoint``, so both artifacts exist and the
    comparison to upstream stays honest.
    """

    def __init__(
        self,
        export_dir: str,
        every_n_epochs: int = 1,
        mirror_dir: str | None = None,
        train_config: DictConfig | None = None,
    ):
        super().__init__()
        self.export_dir = Path(export_dir)
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.mirror_dir = Path(mirror_dir) if mirror_dir else None
        self.train_config = train_config

    @rank_zero_only
    def _export(self, pl_module: pl.LightningModule, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        pl_module.model.export_pretrained(str(destination))

        # Not read by inference -- upstream's from_pretrained leaves the centres
        # as None and its inference path never touches them. Exported anyway so
        # the directory is self-describing and can warm-start further training.
        centers = getattr(pl_module.model, "endpoint_cluster_centers", None)
        if centers is not None:
            torch.save(
                {
                    "endpoint_cluster_centers": centers.detach().cpu(),
                    "centers_fitted": bool(getattr(pl_module.model, "centers_fitted", False)),
                },
                destination / "endpoint_cluster_centers.pt",
            )

        if self.train_config is not None:
            OmegaConf.save(self.train_config, destination / "train_config.yaml")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        self._export(pl_module, self.export_dir)
        if self.mirror_dir is not None:
            self._export(pl_module, self.mirror_dir)
            logger.info("mirrored export to upstream checkpoint_dir: %s", self.mirror_dir)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Guarantee a final export even when max_epochs is not a multiple of
        # every_n_epochs, or when the run stops early.
        self._export(pl_module, self.export_dir)
        if self.mirror_dir is not None:
            self._export(pl_module, self.mirror_dir)


def resolve_mirror_dir(checkpoint_dir_template: str, dataset_name: str) -> str:
    """Expand upstream's ``"./checkpoints/{}/dynamics/"`` template.

    Note the directory upstream reads the simulator from is named ``dynamics``,
    not ``simulator``.
    """
    return os.path.normpath(checkpoint_dir_template.format(dataset_name))
