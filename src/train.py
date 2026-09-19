"""Train the CrowdES locomotion simulator.

    python -m src.train                                  # eth, 64 epochs
    python -m src.train experiment=smoke_gcs             # fast smoke path
    python -m src.train build_cache_only=true data=eth   # build the cache and exit
    python -m src.train ckpt_path=outputs/.../last.ckpt  # resume
"""

from __future__ import annotations

import logging

# Must precede anything that touches upstream.
import src._upstream  # noqa: F401
from src.util.paths import register_resolvers

register_resolvers()

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from src.data.dotdict_bridge import to_crowdes_config  # noqa: E402
from src.data.simulator_datamodule import CrowdESSimulatorDataModule  # noqa: E402
from src.util.export import HFExportCallback, resolve_mirror_dir  # noqa: E402
from src.util.seeding import seed_everything  # noqa: E402

logger = logging.getLogger(__name__)


def build_callbacks(cfg: DictConfig) -> list:
    callbacks = []
    for name, node in (cfg.callbacks or {}).items():
        logger.info("callback: %s", name)
        callbacks.append(hydra.utils.instantiate(node))

    mirror = None
    if cfg.export.mirror_upstream:
        mirror = resolve_mirror_dir(
            cfg.data.crowd_simulator.simulator.checkpoint_dir,
            cfg.data.dataset.dataset_name,
        )
    callbacks.append(
        HFExportCallback(
            export_dir=cfg.export.dir,
            every_n_epochs=cfg.export.every_n_epochs,
            mirror_dir=mirror,
            train_config=cfg,
        )
    )
    return callbacks


@hydra.main(version_base="1.3", config_path="configs", config_name="train_simulator")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info("config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    seed_everything(int(cfg.seed))

    crowdes_cfg = to_crowdes_config(cfg.data)
    datamodule = CrowdESSimulatorDataModule(crowdes_cfg, loader=cfg.data.get("loader"))

    if cfg.build_cache_only:
        datamodule.prepare_data()
        logger.info("cache ready at %s; exiting (build_cache_only=true)", datamodule.cache_path("train"))
        return

    system = hydra.utils.instantiate(cfg.system, _recursive_=False)

    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_kwargs.pop("_target_", None)
    trainer_kwargs["callbacks"] = build_callbacks(cfg)
    trainer_kwargs["logger"] = (
        hydra.utils.instantiate(cfg.logger) if cfg.logger.get("_target_") else False
    )

    import pytorch_lightning as pl

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(system, datamodule=datamodule, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    main()
