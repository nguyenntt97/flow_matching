# `src/` — CrowdES locomotion-simulator parity stack

A PyTorch Lightning + Hydra training and evaluation stack for the **CrowdES
crowd simulator** (Bae et al., CVPR 2025), built by deriving from the vendored
submodule at `third_party/crowdes` rather than forking it.

The goal here is **parity**: reproduce upstream's simulator numbers with code we
own, so the later flow-matching work (`survey/design/`, Subsystem 2) has a
trustworthy baseline and a clean place to plug in. The flow-matching head is
*not* implemented yet; the seam it will use is `src/models/protocol.py`.

## Setup

The existing `ai_riser` env already satisfies everything except pandas, so
cloning it is the fast path:

```bash
conda create -n flowbt --clone ai_riser && conda activate flowbt
pip install "pandas>=2,<3" pytest
python -c "import torch; print(torch.cuda.get_device_capability())"   # expect (12, 0)
```

Or from scratch: `conda env create -f src/environment.yml`.

**One hard pin: `pandas>=2,<3`.** `utils/trajectory.py::groupby_sliding_window`
reads `x.agent_id` inside a `gb.apply(...)` where `agent_id` is the grouping
key. pandas 3.0 stopped passing grouping columns into `apply`, so the group has
no such column. Measured on this machine: **2.3.3 produces correct output**
(windows and `meta_id`s verified) with a `FutureWarning`; **3.0.3 raises
`AttributeError: 'DataFrame' object has no attribute 'agent_id'`**. It fails
loudly rather than corrupting the dataset silently — but it fails, and only
once the build is already under way.

torch must also be a **cu128+** build: the RTX 5090 is sm_120, so upstream's
pinned `torch==2.2.2`/cu121 has no kernels for it. `src/environment.yml`
records the versions actually verified here (torch 2.10.0+cu128,
transformers 4.57.3, pytorch-lightning 2.6.1).

## Running

```bash
# Fast smoke path: gcs has 2 train scenes, so the cache builds in minutes.
python -m src.train experiment=smoke_gcs

# Build the dataset cache and stop -- the first eth build is tens of minutes
# (one A* navmesh search per sample), so make it a deliberate act.
python -m src.train data=eth build_cache_only=true

# The parity run.
python -m src.train experiment=parity_eth

# Resume.
python -m src.train data=eth ckpt_path=outputs/.../checkpoints/last.ckpt

# Agent-level evaluation on the held-out test split.
python -m src.evaluate_agent data=eth ckpt=outputs/.../checkpoints/last.ckpt

# Scene-level benchmark (needs the emitter checkpoints, see below).
python -m src.evaluate_scene data=eth
```

## Layout

| Path | What it is |
|---|---|
| `_upstream.py` | The only place bare `utils.` / `CrowdES.` imports appear. Binds the submodule's two packages into `sys.modules` directly. |
| `data/dotdict_bridge.py` | `DictConfig` → the `DotDict` upstream's classes expect, with a required-key check. |
| `data/simulator_dataset.py` | `ParitySimulatorDataset` — upstream's builder, two overrides. |
| `data/simulator_datamodule.py` | Cache building in `prepare_data`, loaders in `setup`. |
| `models/crowdes_parity.py` | Upstream's model plus durable cluster-centre buffers. |
| `models/protocol.py` | The contract a flow-matching head will satisfy. |
| `systems/simulator_system.py` | The Lightning port of `simulator_trainer.py`. |
| `configs/` | Hydra groups. `data/*.yaml` are transcribed from upstream's three-way yaml split. |
| `tools/diff_config.py` | Verifies that transcription against upstream, at the value level. |

## Five upstream behaviours this code deliberately preserves

Each of these looks like a bug or an oddity. Changing any of them breaks
parity, so they are reproduced on purpose and each is commented at its site.

1. **The environment crop's coordinate ordering.** `SimulatorDataset.__getitem__`
   offsets an `(h, w)` pointer grid by `startpoint[[1, 0]]` and pushes it
   through `world2image`, which reads axis 0 as `x`. The identical code runs in
   `inference_model.py::process_simulator`, so train and inference agree.
   `__getitem__` is inherited verbatim and `tests/parity/test_dataset_parity.py`
   pins it.
2. **`traj_fut is not None`, not `self.training`,** selects the training branch
   and the presence of `.loss`.
3. **Validation runs on the train split.** Upstream constructs
   `SimulatorDataset(config, 'train')` twice and never touches test. Changing
   this would make per-epoch numbers incomparable to upstream's
   `all_results.json`.
4. **AdamW keeps `weight_decay=0.01`.** Upstream passes only `lr`, so torch's
   default applies. Setting it to 0 would be a silent change.
5. **The train loss is logged before NaN-zeroing**, matching upstream's
   `total_loss` accumulation.

## The one upstream behaviour this code fixes

`endpoint_cluster_centers` is a bare tensor attribute on
`CrowdESSimulatorModel` — not a `Parameter`, not a buffer. It is absent from
every state dict, `from_pretrained` returns it as `None`, and the *training*
branch dereferences it. Upstream copes by refitting KMeans before every run,
which is fine for a script and wrong for a resumable trainer: a resumed run
would refit and silently relabel the discrete behaviour states mid-training.

We keep the centres in a `register_buffer` **on the wrapper**, not on `net`.
They ride along in the Lightning `.ckpt` like any other buffer, while
`net.save_pretrained()` still emits exactly upstream's key set — so
`CrowdESFramework` loads our export with no missing/unexpected-key warnings,
and sees `endpoint_cluster_centers is None`, exactly as it does upstream.

## What parity can and cannot be verified against

**Can, with the simulator alone:** per-epoch train loss, `val/ade` and
`val/fde` on the train split with `sampling=False`, the whole metric curve,
the 8×2 cluster centres, parameter count, state-dict keys, and exact batch
shapes and dtypes.

**Needs the emitter too:** everything in `utils/metrics.py::compute_metrics`,
including **collision rate**, because `CrowdESFramework.__init__` loads
emitter_pre, emitter *and* simulator unconditionally. Download the emitter
pair from the upstream release tag `v1.0-model`
(<https://github.com/InhwanBae/Crowd-Behavior-Generation/releases>) into
`checkpoints/<dataset>/{emitter_pre,emitter}/`, and put the simulator in place
with `export.mirror_upstream=true`.

**Cannot be verified bit-exactly, at all:** this GPU forces a torch version
upstream never ran on. The acceptance criterion is final `val_fde` within
~1–2% of upstream, not equality.

**Already verified** (torch 2.10.0+cu128, transformers 4.57.3, PL 2.6.1):
config transcription for all 8 datasets; parameter count and state-dict keys
against upstream; a bit-identical training loss and `preds` for a fixed batch;
the `save_pretrained` → `from_pretrained` round-trip with an identical key set
and `endpoint_cluster_centers is None`; the cluster buffers surviving a
state-dict round-trip without mutating their `control` argument; and that
Lightning restores buffers *before* `on_fit_start`, which is what makes the
no-refit-on-resume guard correct.

The upstream baseline, run around two upstream bugs — `trainval.py` declares
`--model_config` as `type=int` against a string default so every CLI
invocation dies, and the submodule's own CWD cannot see the data:

```bash
PYTHONPATH=third_party/crowdes python -c "
from utils.config import get_config
from CrowdES.simulator.simulator_trainer import main
main(get_config('third_party/crowdes/configs/model/CrowdES_eth.yaml'))"
```

Compare its `all_results.json` to our TensorBoard scalars. Note upstream's
`train_loss` is the **sum** over steps, not the mean — divide by the number of
batches before comparing with `train/loss_epoch`.

## Tests

```bash
pytest tests/parity -m "not slow"   # shim, config bridge, model parity, cluster buffers
pytest tests/parity                 # adds dataset parity (builds the gcs cache)
python -m src.tools.diff_config     # config transcription vs upstream
```

## Three things the submodule assumes that we have to arrange

1. **`dataset_path` must be a real directory.** `BaseDataset` opens
   `join(dataset_path, '..', 'segmentation_classes.json')`, and `open()`
   resolves `..` through the filesystem, so every component has to be
   traversable. Upstream's raw-data dirs (`datasets/ETH-UCY`, `datasets/GCS`, …)
   are not present here, so `dataset_path` points at `datasets/preprocessed`,
   whose parent holds the JSON. `to_crowdes_config` preflights this.
2. **Worker processes need the submodule on `sys.path`.** `sys.modules` is
   process-local, and the build farms `process_scene` out to loky workers that
   import `utils.dataloader.simulator_dataloader` and unpickle a `DotDict`
   defined in `utils.config`. loky copies the parent's `sys.path` but not its
   `sys.modules`, so `_upstream.py` appends the root as well. Without it the
   build dies with `BrokenProcessPool: A task has failed to un-serialize`.
3. **The build worker count has to be overridden, not configured.**
   `SimulatorDataset.__init__` hardcodes `Parallel(n_jobs=256)`, and an
   explicit `n_jobs` beats `joblib.parallel_backend(...)` — verified, so the
   usual context manager does nothing. `capped_build_workers` rebinds
   `Parallel` inside the upstream module for the duration of the build.
   `process_scene` is independent per scene, so this cannot change the result.

## Known upstream issues, worked around rather than patched

The submodule is left untouched. These are handled in `src/`:

- `trainval.py:7` and `utils/preprocess_dataset.py:12` declare `--model_config`
  as `type=int` with a string default — every upstream CLI call fails.
- `scripts/preprocess_dataset.sh` is stale (wrong script name, wrong flags,
  wrong config filenames).
- `configs/model/CrowdES_edin.yaml` sets `dataset_config` to `hotel.yaml`, so a
  plain upstream edin run trains on hotel. `src/configs/data/edin.yaml` points
  at edin and says so.
- `utils/navmesh.py::PathFinderNew.orca_simulation` is broken (a dict is
  `.append`ed to). The working ORCA path is
  `inference_model.py::process_orca_simulator`.
- `utils/preprocessor/*.py` do `from homography import ...`, which only
  resolves with `utils/` itself on `sys.path`. Out of scope — we never
  regenerate the preprocessed data.
