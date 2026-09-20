"""LightningDataModule over the derived CrowdES simulator dataset."""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src._upstream import DotDict
from src.data.simulator_dataset import ParitySimulatorDataset, capped_build_workers
from src.util.fingerprint import build_fingerprint

logger = logging.getLogger(__name__)


class CrowdESSimulatorDataModule(pl.LightningDataModule):
    """Wraps ``ParitySimulatorDataset`` with cache building split out.

    Upstream deliberately validates on the *train* split -- its trainer builds
    ``SimulatorDataset(config, 'train')`` twice and never constructs a test
    loader. We keep that, because changing the validation split would make the
    per-epoch numbers incomparable to upstream's ``all_results.json``.
    """

    def __init__(
        self,
        crowdes_config: DotDict,
        loader: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        self.cfg = crowdes_config
        loader = dict(loader or {})

        sim = self.cfg["crowd_simulator"]["simulator"]
        self.train_batch_size = int(sim["train_batch_size"])
        self.eval_batch_size = int(sim["eval_batch_size"])

        self.num_workers = int(loader.get("num_workers", 8))
        self.pin_memory = bool(loader.get("pin_memory", True))
        self.persistent_workers = bool(loader.get("persistent_workers", True))
        self.val_split = str(loader.get("val_split", "train_same_object"))
        self.build_n_jobs = loader.get("build_n_jobs", None)

        self.fingerprint = build_fingerprint(sim)
        self.cache_dir = os.path.join(self.cfg["dataset"]["dataset_preprocessed_path"], "cache")

        # The cache lives on a local mount, so every node needs its own copy.
        self.prepare_data_per_node = True

        self.train_set: ParitySimulatorDataset | None = None
        self.val_set: ParitySimulatorDataset | None = None

    # ------------------------------------------------------------------ paths
    def cache_path(self, phase: str) -> str:
        return os.path.join(self.cache_dir, f"{phase}_simulator.{self.fingerprint}.pickle")

    # ------------------------------------------------------------- lightning
    def prepare_data(self) -> None:
        """Build the cache once per node. Must not assign to ``self.*``.

        This is the slow part: one A* navmesh search per sample via
        ``get_control_point``, farmed out by upstream with a hardcoded
        ``joblib.Parallel(n_jobs=256)``. See ``capped_build_workers`` for why
        that needs overriding and why doing so cannot change the result.
        """
        path = self.cache_path("train")
        if os.path.exists(path):
            logger.info("dataset cache present: %s", path)
            return

        n_jobs = self.build_n_jobs or (os.cpu_count() or 1)
        logger.info("building dataset cache at %s (n_jobs=%d); this can take a while", path, n_jobs)
        with capped_build_workers(n_jobs):
            ParitySimulatorDataset(self.cfg, "train", cache_path=path)

    def setup(self, stage: str | None = None) -> None:
        if stage not in (None, "fit", "validate"):
            return
        if self.train_set is not None:
            return

        path = self.cache_path("train")
        self.train_set = ParitySimulatorDataset(self.cfg, "train", cache_path=path)

        if self.val_split == "train_same_object":
            # Upstream builds a second identical object; both would come from
            # the same cache file and be value-identical. Sharing one halves
            # the resident memory and changes nothing.
            self.val_set = self.train_set
        elif self.val_split == "train_copy":
            self.val_set = ParitySimulatorDataset(self.cfg, "train", cache_path=path)
        else:
            raise ValueError(
                f"unknown val_split {self.val_split!r}; expected 'train_same_object' or 'train_copy'"
            )

        logger.info(
            "dataset ready: %d scenes, %d samples, env_dim=%d",
            len(self.train_set.scene_list),
            len(self.train_set),
            self.train_set.env_dim,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        # num_workers=0: __getitem__ is an index plus a fancy-index crop, and
        # forking workers would make a second copy of a multi-GB in-RAM dataset
        # (COW does not save you -- refcount writes dirty the pages).
        return DataLoader(
            self.val_set,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
