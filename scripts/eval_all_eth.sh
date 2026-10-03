#!/usr/bin/env bash
# Evaluate CrowdES, the flow teacher and Flow2BT on eth, agent and scene level.
#
#   bash scripts/eval_all_eth.sh            # everything
#   bash scripts/eval_all_eth.sh scene      # one section: agent | openloop | scene | timing | viz
#
# TRIALS=20 matches CrowdES's protocol; set TRIALS=5 for a quicker pass.
# Every run writes to outputs/<name>/<timestamp>/ with the exact overrides in
# .hydra/overrides.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."

TRIALS=${TRIALS:-20}
SECTION=${1:-all}

CROWDES=checkpoints/eth/simulator
TEACHER=outputs/flow_eth/20260920-072219/checkpoints/epoch063-minfde0.0529.ckpt
GT=outputs/induce_ground_truth_eth/20260930-150453/bundle.pkl      # no teacher (ablation)
R4=outputs/induce_flow_r4_eth/20260930-150527/bundle.pkl           # Flow2BT, 4096 states x 4
R16=outputs/induce_flow_r16_eth/20260930-150556/bundle.pkl         # Flow2BT, 1024 states x 16

run() { echo; echo ">>> $*"; "$@"; }

# ---- 1. Agent level: ADE/FDE, minADE20/minFDE20, 3-round rollout (test split)
if [[ $SECTION == all || $SECTION == agent ]]; then
  for seed in 0 1 2; do
    run python -m src.evaluate_agent data=eth num_samples=20 seed=$seed \
        ckpt=$CROWDES name=cmp_agent_eth_crowdes_s$seed
    run python -m src.evaluate_agent data=eth num_samples=20 seed=$seed \
        ckpt=$TEACHER name=cmp_agent_eth_flow_s$seed
  done
fi

# ---- 2. Open-loop tree: routed / oracle prototype and DMP ADE/FDE (test split)
if [[ $SECTION == all || $SECTION == openloop ]]; then
  for tag in gt:$GT r4:$R4 r16:$R16; do
    run python -m src.evaluate_bt_openloop data=eth bundle=${tag#*:} name=ol_${tag%%:*}
  done
fi

# ---- 3. Scene level: collisions + realism (seq_eth, TRIALS trials, CBF@0.45)
if [[ $SECTION == all || $SECTION == scene ]]; then
  run python -m src.evaluate_scene data=eth trials=$TRIALS \
      data.crowd_simulator.simulator.checkpoint_dir=./$CROWDES/ \
      label=CrowdES-pretrained name=scene_crowdes

  FLOW2BT="python -m src.evaluate_flow2bt data=eth trials=$TRIALS d_min_sweep=[0.45]"
  # Safety-filter-only floor: what the CBF achieves without any tree.
  run $FLOW2BT runtime.controller=waypoint name=scene_waypoint_cbf
  # Flow2BT proper (trees distilled from the flow teacher).
  run $FLOW2BT runtime.controller=bt runtime.bt_bundle=$R16 name=scene_flow2bt_r16
  run $FLOW2BT runtime.controller=bt runtime.bt_bundle=$R4  name=scene_flow2bt_r4
  # Ablation: same pipeline, tree induced from ground truth (no teacher).
  run $FLOW2BT runtime.controller=bt runtime.bt_bundle=$GT  name=scene_gt_tree
fi

# ---- 4. Inference timing (1 trial each; not part of "all")
# Agent level times every forward at the eval batch size; set batch_size=16 to
# time at a crowd-like batch. Scene level times the simulator forward per 2 s
# chunk (CrowdES, flow teacher) against one BT tick, in the same scene.
# runtime.substeps=20 runs the BT at 100 Hz (dt = 10 ms); the default is 4 (20 Hz).
if [[ $SECTION == timing ]]; then
  run python -m src.evaluate_agent data=eth num_samples=1 rollout.enabled=false timing=true \
      ckpt=$CROWDES name=timing_agent_crowdes
  run python -m src.evaluate_agent data=eth num_samples=1 rollout.enabled=false timing=true \
      ckpt=$TEACHER name=timing_agent_flow
  run python -m src.evaluate_scene data=eth trials=1 timing=true \
      data.crowd_simulator.simulator.checkpoint_dir=./$CROWDES/ \
      label=CrowdES name=timing_scene_crowdes
  run python -m src.evaluate_scene data=eth trials=1 timing=true \
      data.crowd_simulator.simulator.checkpoint_dir=./$CROWDES/ \
      simulator_ckpt=$TEACHER label=FlowTeacher name=timing_scene_flow
  run python -m src.evaluate_flow2bt data=eth trials=1 d_min_sweep=[0.45] timing=true \
      runtime.controller=bt runtime.bt_bundle=$R16 runtime.substeps=20 name=timing_scene_flow2bt_100hz
fi

# ---- 5. Videos (1 trial each)
if [[ $SECTION == viz ]]; then
  run python -m src.evaluate_agent data=eth num_samples=5 viz=true viz_limit=20 \
      ckpt=$CROWDES name=viz_agent_crowdes
  run python -m src.evaluate_agent data=eth num_samples=5 viz=true viz_limit=20 \
      ckpt=$TEACHER name=viz_agent_flow
  run python -m src.evaluate_scene data=eth trials=1 viz=true \
      data.crowd_simulator.simulator.checkpoint_dir=./$CROWDES/ \
      label=CrowdES-pretrained name=viz_scene_crowdes
  run python -m src.evaluate_flow2bt data=eth trials=1 d_min_sweep=[0.45] viz=true \
      runtime.controller=bt runtime.bt_bundle=$R16 name=viz_scene_flow2bt_r16
fi
