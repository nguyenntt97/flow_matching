"""Compare our Hydra data configs against upstream's merged ``get_config()``.

    python -m src.tools.diff_config              # all datasets
    python -m src.tools.diff_config eth gcs      # just these

This is the transcription check: our ``src/configs/data/*.yaml`` were written
by hand from upstream's three-way yaml split, and this verifies, at the value
level, that nothing was mistyped and that upstream has not changed under us.
Comparing values beats regenerating and byte-diffing text -- it survives
formatting differences and reports exactly which key drifted.

Expected differences are allowlisted below: every path key, because we made
them absolute and repo-anchored on purpose, plus upstream's edin/hotel mixup.
"""

from __future__ import annotations

import sys
from pathlib import Path

import src._upstream as up
from src.util.paths import register_resolvers

register_resolvers()

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.data.dotdict_bridge import UPSTREAM_SUBTREES, to_crowdes_config  # noqa: E402

DATASETS = ("eth", "hotel", "univ", "zara1", "zara2", "sdd", "gcs", "edin")

#: Leaf keys whose divergence is intended.
PATH_KEYS = {
    "dataset_path",
    "dataset_preprocessed_path",
    "checkpoint_dir",
    "cache_dir",
    "model_pretrained",
}

#: Keys upstream has that we deliberately drop (consumed only by the
#: preprocessor or by upstream's own config plumbing).
DROPPED_KEYS = {
    "dataset_train",
    "dataset_test",
    "use_logger",
    "logger_type",
}


def flatten(node, prefix=""):
    out = {}
    if isinstance(node, dict):
        for key, value in node.items():
            out.update(flatten(value, f"{prefix}{key}."))
    else:
        out[prefix.rstrip(".")] = node
    return out


def upstream_config(dataset: str):
    """Load upstream's merged config without relying on the CWD.

    ``get_config`` resolves whatever ``dataset_config`` / ``trainer_config``
    the model yaml declares, and those are written ``./configs/...`` --
    relative to a CWD upstream assumes is the submodule root. We never chdir,
    so both have to be passed in absolute. The trainer file differs per
    dataset, so read the name the model yaml declares rather than guessing it.
    """
    root = up.CROWDES_ROOT
    model_path = root / "configs" / "model" / f"CrowdES_{dataset}.yaml"

    declared = up.load_config(str(model_path))
    trainer_name = Path(declared.trainer_config).name

    return up.get_config(
        str(model_path),
        # Upstream's CrowdES_edin.yaml declares hotel.yaml here, which looks
        # like a copy-paste slip. We compare against the dataset the name says.
        dataset_config=str(root / "configs" / "dataset" / f"{dataset}.yaml"),
        trainer_config=str(root / "configs" / "trainer" / trainer_name),
    )


def ours(dataset: str):
    config_dir = str((up.REPO_ROOT / "src" / "configs").resolve())
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="train_simulator", overrides=[f"data={dataset}"])
    return to_crowdes_config(cfg.data)


def compare(dataset: str) -> int:
    theirs = flatten({k: v for k, v in upstream_config(dataset).items() if k in UPSTREAM_SUBTREES})
    mine = flatten(dict(ours(dataset)))

    problems = 0
    for key in sorted(set(theirs) | set(mine)):
        leaf = key.rsplit(".", 1)[-1]
        if leaf in PATH_KEYS or leaf in DROPPED_KEYS:
            continue
        their_value = theirs.get(key, "<absent>")
        my_value = mine.get(key, "<absent>")
        if isinstance(their_value, (list, tuple)) and isinstance(my_value, (list, tuple)):
            same = list(their_value) == list(my_value)
        else:
            same = their_value == my_value
        if not same:
            print(f"  DIFF {key}: upstream={their_value!r} ours={my_value!r}")
            problems += 1

    print(f"{dataset}: {'OK' if problems == 0 else f'{problems} difference(s)'}")
    return problems


def main(argv: list[str]) -> int:
    datasets = argv or list(DATASETS)
    total = 0
    for dataset in datasets:
        if dataset == "edin":
            print(
                "edin: upstream's model yaml points dataset_config at hotel.yaml, so its "
                "merged dataset block is hotel's. Comparing against dataset/edin.yaml instead."
            )
        total += compare(dataset)
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
