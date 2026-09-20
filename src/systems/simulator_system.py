"""LightningModule port of ``CrowdES/simulator/simulator_trainer.py``.

Parity notes are inline. The short version of what upstream does that is easy
to get wrong: it selects the training branch on ``traj_fut is not None`` rather
than ``self.training``; it logs the loss *before* zeroing NaNs; it passes only
``lr`` to AdamW (so weight_decay is torch's 0.01 default, not 0); and it
validates on the train split with ``sampling=False``.
"""

from __future__ import annotations

import logging
import math

import hydra
import pytorch_lightning as pl
import torch
import transformers
from lightning_utilities.core.rank_zero import rank_zero_info
from torchmetrics import MeanMetric

logger = logging.getLogger(__name__)


class SimulatorLitModule(pl.LightningModule):
    def __init__(
        self,
        model_cfg,
        learning_rate: float = 1e-4,
        lr_scheduler_type: str = "polynomial",
        num_warmup_steps: int = 0,
        hist_dropout_full_p: float = 0.1,
        hist_dropout_prefix_p: float = 0.1,
        assert_scheduler_horizon: bool = True,
        val_num_samples: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = hydra.utils.instantiate(model_cfg)

        self.val_ade = MeanMetric()
        self.val_fde = MeanMetric()

        # minADE_K / minFDE_K, off by default so the parity path is untouched.
        # A generative head (the flow teacher) samples, so ADE/FDE scores one
        # draw against the ground truth and is worse than an L1 regression by
        # construction. These are the statistic to select such a head on.
        if val_num_samples > 0:
            self.val_min_ade = MeanMetric()
            self.val_min_fde = MeanMetric()

        self._restored_from_ckpt = False

    # ---------------------------------------------------------- augmentation
    def _history_dropout(self, traj_hist: torch.Tensor) -> torch.Tensor:
        """Vectorised form of upstream's two train-loop history dropouts.

        Upstream draws ``rand(B)``, ``rand(B)``, ``randint(0, T, (B,))`` on the
        *CPU* global generator in that order, then zeroes whole histories and
        random prefixes respectively. We keep the device and the order: drawing
        on CUDA instead would fork the random stream -- statistically the same,
        but no longer step-comparable against an upstream baseline run.
        """
        batch, horizon, _ = traj_hist.shape
        device = traj_hist.device
        traj_hist = traj_hist.clone()  # never mutate the batch in place

        drop_full = (torch.rand(batch) < self.hparams.hist_dropout_full_p).to(device)
        traj_hist[drop_full] = 0

        drop_prefix = (torch.rand(batch) < self.hparams.hist_dropout_prefix_p).to(device)
        cut = torch.randint(0, horizon, (batch,)).to(device)
        steps = torch.arange(horizon, device=device).unsqueeze(0)
        keep = ~(drop_prefix.unsqueeze(1) & (steps < cut.unsqueeze(1)))
        return traj_hist * keep.unsqueeze(-1)

    # ----------------------------------------------------------------- hooks
    def on_load_checkpoint(self, checkpoint) -> None:
        # Defensive: buffers should already be restored by the time
        # on_fit_start runs, but this flag does not depend on that ordering.
        self._restored_from_ckpt = True

    def on_fit_start(self) -> None:
        datamodule = self.trainer.datamodule
        train_set = datamodule.train_set

        expected_env_dim = self.model.net.config.env_dim
        if train_set.env_dim != expected_env_dim:
            raise ValueError(
                f"env_dim mismatch: dataset produces {train_set.env_dim} channels, "
                f"model was built for {expected_env_dim}. Check "
                "crowd_simulator.simulator.environment_types against system.model_cfg.env_dim."
            )

        if self.hparams.assert_scheduler_horizon:
            self._check_scheduler_horizon()

        if not getattr(self.model, "needs_endpoint_clusters", False):
            return

        if bool(self.model.centers_fitted):
            rank_zero_info("endpoint cluster centres restored from checkpoint; not refitting")
            return
        if self._restored_from_ckpt:
            raise RuntimeError(
                "resumed from a checkpoint whose endpoint cluster centres were never fitted; "
                "refitting now would relabel the discrete behaviour states mid-training"
            )

        self._fit_endpoint_clusters(train_set)

    def on_train_start(self) -> None:
        if getattr(self.model, "needs_endpoint_clusters", False):
            if not bool(self.model.centers_fitted):
                raise RuntimeError("endpoint cluster centres were never fitted")

    def _fit_endpoint_clusters(self, train_set) -> None:
        """Reproduce upstream's pre-training clustering step exactly.

        Upstream concatenates every scene's final future point and control
        point, mirrors both in x, and fits KMeans(k=latent_dim). Scene order
        comes from ``scene_list``, which is sorted, so the concatenation is
        deterministic.
        """
        mirror = torch.tensor([-1.0, 1.0])
        endpoints = torch.cat(
            [train_set.traj_pred_norm_scene[scene][:, -1, :] for scene in train_set.scene_list], dim=0
        )
        controls = torch.cat(
            [train_set.control_point_scene[scene] for scene in train_set.scene_list], dim=0
        )
        endpoints = torch.cat([endpoints, endpoints * mirror], dim=0)
        controls = torch.cat([controls, controls * mirror], dim=0)

        self.model.fit_endpoint_clusters(endpoints, controls)

        # KMeans(random_state=0) is deterministic given identical input, but
        # broadcast anyway so ranks can never diverge on thread-count effects.
        self.model.endpoint_cluster_centers.copy_(
            self.trainer.strategy.broadcast(self.model.endpoint_cluster_centers, src=0)
        )
        rank_zero_info(
            "fitted %d endpoint cluster centres from %d endpoints:\n%s",
            self.model.endpoint_cluster_centers.shape[0],
            endpoints.shape[0],
            self.model.endpoint_cluster_centers,
        )

    def _check_scheduler_horizon(self) -> None:
        """Upstream: ``num_train_epochs * ceil(len(loader) / accum)``.

        Lightning's ``estimated_stepping_batches`` matches that for
        devices=1/accum=1/drop_last=False. If it does not, the polynomial decay
        reaches zero at a different point and the LR curve silently diverges.
        """
        trainer = self.trainer
        accum = max(1, trainer.accumulate_grad_batches)
        batches = len(trainer.datamodule.train_dataloader())
        upstream = trainer.max_epochs * math.ceil(batches / accum)
        actual = trainer.estimated_stepping_batches
        if upstream != actual:
            logger.warning(
                "scheduler horizon differs from upstream: upstream=%d, lightning=%d "
                "(devices=%s, accum=%d). The polynomial LR decay endpoint will differ.",
                upstream,
                actual,
                trainer.num_devices,
                accum,
            )
        else:
            rank_zero_info("scheduler horizon matches upstream: %d steps", actual)

    # ----------------------------------------------------------------- steps
    def training_step(self, batch, batch_idx):
        traj_hist = self._history_dropout(batch["traj_hist"])

        # Positional, in upstream's order. control is cloned because
        # normalize_rotation_scale writes into near-zero rows in place -- inert
        # within one forward, but the batch tensor may alias dataset storage
        # when num_workers=0.
        outputs = self.model(
            traj_hist,
            batch["traj_fut"],
            batch["goal"],
            batch["attr"],
            batch["control"].clone(),
            batch["neighbor"],
            batch["environment"],
        )
        loss = outputs.loss

        # Logged before zeroing, matching upstream's total_loss accumulation.
        self.log(
            "train/loss",
            loss.detach(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=traj_hist.shape[0],
        )
        # Out-of-place equivalent of upstream's `loss[isnan | isinf] = 0`.
        return loss.masked_fill(torch.isnan(loss) | torch.isinf(loss), 0.0)

    def validation_step(self, batch, batch_idx):
        outputs = self.model(
            batch["traj_hist"],
            None,
            batch["goal"],
            batch["attr"],
            batch["control"].clone(),
            batch["neighbor"],
            batch["environment"],
            sampling=False,  # argmax latent, as upstream's validation does
        )
        distance = torch.linalg.norm(outputs.preds - batch["traj_fut"], dim=-1)  # (B, T_fut), metres
        # Feeding per-sample vectors makes MeanMetric sum values and count
        # elements, giving upstream's concatenate-then-mean number exactly,
        # plus correct DDP reduction.
        self.val_ade.update(distance.mean(dim=-1))
        self.val_fde.update(distance[:, -1])

        num_samples = self.hparams.val_num_samples
        if num_samples > 0:
            # Same arithmetic as src/evaluate_agent.py::evaluate_split, so the
            # per-epoch number and the final evaluation are the same statistic.
            best_ade = best_fde = None
            for _ in range(num_samples):
                sampled = self.model(
                    batch["traj_hist"],
                    None,
                    batch["goal"],
                    batch["attr"],
                    batch["control"].clone(),
                    batch["neighbor"],
                    batch["environment"],
                    sampling=True,
                )
                d = torch.linalg.norm(sampled.preds - batch["traj_fut"], dim=-1)
                ade, fde = d.mean(dim=-1), d[:, -1]
                best_ade = ade if best_ade is None else torch.minimum(best_ade, ade)
                best_fde = fde if best_fde is None else torch.minimum(best_fde, fde)
            self.val_min_ade.update(best_ade)
            self.val_min_fde.update(best_fde)

    def on_validation_epoch_end(self) -> None:
        self.log("val/ade", self.val_ade.compute(), prog_bar=True, sync_dist=True)
        self.log("val/fde", self.val_fde.compute(), prog_bar=True, sync_dist=True)
        self.val_ade.reset()
        self.val_fde.reset()

        if self.hparams.val_num_samples > 0:
            self.log("val/min_ade", self.val_min_ade.compute(), prog_bar=True, sync_dist=True)
            self.log("val/min_fde", self.val_min_fde.compute(), prog_bar=True, sync_dist=True)
            self.val_min_ade.reset()
            self.val_min_fde.reset()

    # ------------------------------------------------------------ optimisers
    def configure_optimizers(self):
        # Upstream passes only lr, so weight_decay stays at torch's AdamW
        # default of 0.01. Do not "helpfully" set it to 0.
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = transformers.get_scheduler(
            name=self.hparams.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
