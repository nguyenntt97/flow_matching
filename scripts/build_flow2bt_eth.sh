#!/usr/bin/env bash
# Rebuild the Flow2BT artefacts on eth: flow teacher -> induced trees.
#
#   bash scripts/build_flow2bt_eth.sh            # teacher + all trees
#   bash scripts/build_flow2bt_eth.sh induce     # trees only, from an existing teacher
#   TEACHER=path/to.ckpt bash scripts/build_flow2bt_eth.sh induce
#
# The teacher run (64 epochs, batch 2048) took ~3 h 40 min on the RTX 5090.
# Induction takes about a minute per tree.
#
# Outputs get new timestamps, so the paths differ from the 2026-09-30 ones.
# scripts/eval_all_eth.sh picks up the newest of each automatically.
# Expect close but not bit-identical numbers: GPU training is not deterministic.
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE=${1:-all}
latest() { ls -td $1 2>/dev/null | head -1; }

# ---- 1. Flow teacher (Subsystem 2). Needs datasets/preprocessed/eth/cache/.
if [[ $STAGE == all || $STAGE == teacher ]]; then
  python -m src.train experiment=flow_eth
fi

# Best checkpoint by val/min_fde (the filename carries it); last.ckpt as fallback.
if [[ -z ${TEACHER:-} ]]; then
  RUN=$(latest "outputs/flow_eth/*/")
  [[ -n $RUN ]] || { echo "no outputs/flow_eth run; train the teacher first" >&2; exit 1; }
  TEACHER=$(python - "$RUN" <<'EOF'
import re, sys
from pathlib import Path
ckpts = list(Path(sys.argv[1], "checkpoints").glob("epoch*-minfde*.ckpt"))
score = lambda p: float(re.search(r"minfde([0-9.]+)\.ckpt$", p.name).group(1))
print(min(ckpts, key=score) if ckpts else Path(sys.argv[1], "checkpoints", "last.ckpt"))
EOF
)
fi
echo "teacher: $TEACHER"

# ---- 2. Induction (Subsystems 3-6): ensemble -> dendrogram -> guards + DMPs -> tree
if [[ $STAGE == all || $STAGE == induce ]]; then
  # Flow2BT proper. r16 is the configured default (rollouts_per_state=16).
  python -m src.induce_flow2bt data=eth source=flow ckpt="$TEACHER" \
      num_states=1024 rollouts_per_state=16 name=induce_flow_r16_eth
  python -m src.induce_flow2bt data=eth source=flow ckpt="$TEACHER" \
      num_states=4096 rollouts_per_state=4 name=induce_flow_r4_eth
  python -m src.induce_flow2bt data=eth source=flow ckpt="$TEACHER" \
      num_states=4096 rollouts_per_state=1 name=induce_flow_r1_eth
  # Ablation: same pipeline, no teacher.
  python -m src.induce_flow2bt data=eth source=ground_truth num_states=4096 \
      name=induce_ground_truth_eth

  echo
  for d in induce_flow_r16_eth induce_flow_r4_eth induce_flow_r1_eth induce_ground_truth_eth; do
    echo "$d: $(latest "outputs/$d/*/")bundle.pkl"
  done
fi
