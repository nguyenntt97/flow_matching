# Flow2BT: what the implementation measured

**Status:** Subsystems 1–7a implemented in `src/flow2bt/` and `src/runtime/`, 158 tests green.
**Anchor task:** CrowdES locomotion simulator, ETH (`seq_eth` test scene).
**Scope of this document:** every number below was measured in this repo on this
machine. Where a design claim did not survive contact with the code or the data,
that is recorded here rather than worked around quietly.

---

## 1. Corrections to the design documents

### 1.1 The chunk is 2 s, not 4 s

`third_party/crowdes/configs/model/CrowdES_eth.yaml` sets `future_length: 10` at
`simulator_fps: 5`. Seven of the eight dataset configs use 10; **gcs uses 50
(10 s)**. The design's "T_f = 20 frames at 5 fps" appears nowhere.

Consequences: the reactivity gap is 200×, not 400×, and CrowdES's decision rate
is 0.5 Hz, not 0.25 Hz. Also, `experiment=smoke_gcs` — the repo's fast path —
exercises a 10 s horizon, so it does not smoke-test the shape every other
dataset uses. `experiment=smoke_eth` was added for that.

### 1.2 The CrowdES simulator is one MLP forward pass

`CrowdES/simulator/simulator_model.py:176-180` reshapes a single decoder output
to `(B, T_fut, 2)`. There is no diffusion and no ODE solve; the 50-step DDIM
belongs to the *emitter*. Multimodality comes only from the one-hot `latent`
over B=8 KMeans clusters.

So the design's "35–50 ms per agent on GPU → 400× latency reduction" does not
describe this baseline. The baseline is already fast and batches every agent in
the scene into one call. Any latency claim has to be measured against it.

### 1.3 Collisions are scored at 0.2 m, and ground truth is already clean there

`utils/metrics.py:457-481` counts agent-frames where any other agent is within
**0.2 m centre to centre**, over total agent-frame rows, for the generated
scenario only — `compute_metrics` never computes it for ground truth.
`src/tools/anchors.py` does:

| threshold | GT collision rate (eth test, 184,167 agent-frames) |
|---|---|
| 0.10 m | 0.0000 |
| **0.20 m** (benchmark) | **0.0001** |
| 0.30 m | 0.0003 |
| 0.45 m | 0.0024 |
| **0.70 m** (design's `d_min`) | **0.0866** |

Nearest-neighbour distance across all agent-frames: p1 = 0.526 m, p5 = 0.635 m,
p25 = 0.859 m, p50 = 1.169 m.

**This refutes `d_min = 0.7 m`.** Real pedestrians are inside it on 8.7% of
agent-frames. A filter enforcing 0.7 m makes the crowd obey a spacing the data
does not, and the realism metrics are where that is paid for. Ground truth is
essentially collision-free at 0.2 m, so targeting `C_R → 0` *there* matches
reality instead of distorting it — the design's goal is right, its parameter is
not.

### 1.4 A fourth upstream defect

`src/README.md` records three. A fourth: `CrowdES/evaluate.py:52` nests
same-type quotes inside an f-string expression, which is Python 3.12+ only
(PEP 701). On this repo's Python 3.11 the module cannot be imported at all, so
the published evaluation entry point does not run. `src/eval_loop.py` is a
faithful port (TRIALS=20, `seed=trial`, same `compute_metrics` call).

---

## 2. Measured baselines

`src/evaluate_agent.py` gained the ability to load an HF export, so upstream's
**released** simulator is scored by exactly the same code as ours — no
retraining needed for a baseline.

Agent-level, eth test split, 29,275 samples:

| | ADE | FDE | minADE₂₀ | minFDE₂₀ |
|---|---|---|---|---|
| CrowdES (released checkpoint) | 0.2896 | 0.5659 | 0.2374 | 0.4556 |

Scene-level, `seq_eth`, **20 trials** (upstream's `TRIALS`, marked "DO NOT
CHANGE THIS!"):

| | Collision | Density | Kinematics | DTW | Diversity |
|---|---|---|---|---|---|
| Ground truth | 0.0001 | — | — | — | — |
| CrowdES (released) | **0.00781 ± 0.00193** | 0.01801 | 0.33990 | 1.62152 | 0.19921 |

Per-trial collision rate: min 0.00517, median 0.00717, max 0.01137. It generates
277 agents on average against 406 in the ground truth.

CrowdES sits ~78× above ground truth on collisions — the gap Flow2BT targets.
Note it is **~4x below the design's quoted 2.5–3.2%**, so the headline
improvement available is correspondingly smaller than the design assumes.

---

## 3. The headline result: where the collisions actually come from

Running the reactive runtime with the deliberately naive `WaypointController`
(follow the A* waypoint at preferred pace, **no** avoidance) plus the CBF at
`d_min = 0.45`:

| | Collision | raw Collision | Density | Kinematics | DTW | Diversity |
|---|---|---|---|---|---|---|
| CrowdES | 0.00627 | — | 0.01772 | 0.34462 | 1.60840 | 0.21191 |
| waypoint + CBF@0.45 | 0.00317 | **0.00000** | 0.01951 | 0.91872 | 2.22979 | 0.18677 |

`raw Collision` is the collision rate on the simulated state — what the filter
actually controls. It is **exactly zero**. The reported number is introduced
*after* the filter finishes, by upstream's post-processing. A full 2x2x2
ablation over the three steps that can move an agent isolates which one:

| snap | Kalman | dense | reported Collision | raw | Kinematics |
|---|---|---|---|---|---|
| on  | on  | on  | 0.00317 | 0.00000 | 0.919 |
| on  | on  | off | 0.00259 | 0.00000 | 0.451 |
| on  | off | on  | 0.00317 | 0.00000 | 0.919 |
| on  | off | off | 0.00317 | 0.00000 | 0.919 |
| **off** | on  | on  | **0.00000** | 0.00000 | 0.497 |
| **off** | on  | off | **0.00000** | 0.00000 | 0.494 |
| **off** | off | on  | **0.00000** | 0.00000 | 0.497 |
| **off** | off | off | **0.00000** | 0.00000 | 0.497 |

**The walkable-mask snap is the sole and complete cause.** Kalman smoothing and
the 5→25 fps interpolation contribute nothing to the collision rate in any
combination. `batched_nearest_nonzero_idx_kdtree` (`inference_model.py:422`)
moves every agent to the nearest walkable pixel, and two agents near a
non-traversable boundary get moved onto the *same* pixel.

Kalman smoothing does matter, but for realism: it is worth roughly 2x on the
Kinematics metric (0.451 with, 0.919 without). That is why `dense_output`
defaults to **False** — emitting the simulated path directly bypasses the
smoother, and buys nothing because raw collisions are already zero.

Supporting evidence, from the runtime's own diagnostics over 164,980
agent-ticks:

* `min_separation` = 0.29382 m, and `spawn_min_separation` = **0.29382 m** —
  identical. The global closest approach of the entire run *is* the separation
  at which the emitter spawned an agent. Four agents were born inside `d_min`,
  and a barrier certifies forward invariance only from a safe state.
* `untracked_close_pairs` = 0 — the `interaction_max_num_agents = 4` cap never
  dropped a close pair.
* `relaxed_agent_ticks` = 125 (0.08%) — infeasibility is rare.

**So the design's "C_R = 0.0%" is achievable and was achieved** on the state the
controller owns. The reported metric cannot reach zero while upstream's
post-processing is in the loop, and no controller change can fix that. Two
things outside the design's scope would be needed: spawn admission control, and
a post-processing pipeline that preserves separation.

The cost is visible and must not be buried: **Kinematics degrades 0.345 → 0.919
and DTW 1.61 → 2.23.** The naive waypoint controller with a hard safety filter
produces jerky, unrealistic motion (Travel Acceleration 0.349 → 2.229). That is
the floor the behavior tree has to beat, and it is the right comparison — not
CrowdES — for judging what the *tree* contributes as opposed to the *filter*.

---

## 4. Findings that challenge the design

### 4.1 DMP leaves cannot be integrated at the simulator's own frame rate

With the design's spring-damper and `K = 100`, `tau = 2 s`, the damping
coefficient is `2*sqrt(K)/tau = 10 s⁻¹`, and semi-implicit Euler needs
`dt < 0.2 s` — exactly the simulator's step. Measured: integrating at 0.2 s
diverges to tens of metres within one chunk; at 10 ms it converges to the goal.

This is an *independent, numerical* argument for the design's fine tick,
separate from reactivity, and it is not made in the design documents.

Residual accuracy floor: ~1.7 cm over a 2.6 m movement, from Euler being first
order. It does not improve as the forcing fit improves. Well inside the 0.2 m
the benchmark scores at.

### 4.2 Basis count must track sample count

A leaf's demonstration at 5 fps over a 2 s chunk is **11 points**. Fitting 20
Gaussian bases to 11 points is under-determined and ridge at 1e-6 does not
rescue it: replay error rose from 2.2 cm at P=10 to **8.9 cm at P=20**.
`fit_dmp` now resamples onto the integration grid first, after which error falls
monotonically to the integrator floor.

### 4.3 The CBF slack weight is a safety parameter, and must be large

Design 07 eq. (1.1.3) includes a slack term but says nothing about its weight.
Measured on a two-agent head-on at `d_min = 0.7`:

| weight | min separation | relaxation |
|---|---|---|
| 1 | 0.003 m | 1.81 |
| 10 | 0.621 m | 0.209 |
| 100 | 0.693 m | 0.019 |
| 1e4 | **0.700 m** | 0.0002 |

Two things that look like reasons to lower it are not: 60 and 400 solver sweeps
give *identical* separation at every weight, and the slow-converging quantity is
the slack variable, not the control. Because ξ is therefore a poor infeasibility
signal at the weight safety requires, infeasibility is reported as the
constraint **residual** instead.

### 4.4 Decentralised CBF without responsibility sharing stalls

Design 07 eq. (1.1.1) leaves `u_j` on the right-hand side, i.e. each agent
solves as if its neighbour were passive. With both agents doing that, both apply
the full correction and over-brake. `responsibility = 0.5` splits it; summing
the two agents' constraints recovers the joint constraint exactly, because every
non-`u` term is symmetric in `i` and `j`. Measured: sharing yields strictly more
forward progress on the head-on.

### 4.5 The obstacle barrier as written cannot constrain an acceleration

Design 07 eq. (1.1.2) gives `grad(h)ᵀ v + α h ≥ 0`, which contains no `u`. Under
double-integrator dynamics `h_obs` is relative degree 2 like the pairwise
barrier and needs the same higher-order treatment.

### 4.6 Bifurcations are only partly decidable from observable state

This is the most consequential finding for the design's Subsystem 3 → 4 handoff.

Inducing from 4,096 ground-truth futures on eth, cut to CrowdES's B=8:

* agreement with CrowdES's own B=8 KMeans modes: **ARI = 0.448** — the
  hierarchical induction finds a partly different partition, so the claim that
  it recovers the same structure is not supported, and neither is the claim that
  a flat KMeans is simply worse. It is *different*, and that needs arguing on
  merits.
* **tree routing fidelity = 54.1%** — the assembled tree reproduces only 54% of
  the leaf assignment the dendrogram produced. On synthetic data where the
  initial state determines the mode, the same code achieves >95%.
* 1 of 7 guards fails to separate at all (cross-validated accuracy 0.574); the
  rest range 0.70–0.955, and every one is dominated by the agent's own current
  `speed`.
* leaf dispersion 0.22–0.58 m RMS. Design 05's premise is that a leaf bundle is
  unimodal enough for one DMP; at 0.58 m over a ~2.6 m movement, several leaves
  are not.

**The structural point:** a bifurcation in future-trajectory space is only
reproducible by a reactive guard to the extent that it is predictable from what
the agent can observe *now*. Trajectories can diverge because of something that
happens later, and no hyperplane over present state can recover that. The
measured 54% is an upper bound on what any reactive Behavior Tree built this way
can reproduce of the offline structure — and it is a property of the data, not
of the fitting.

`src/induce_flow2bt.py` keeps weak guards and reports them rather than filtering
them out, because this is the finding, not noise.

---

## 5. What the caching bought

The runtime plans A* **once per agent** and projects onto the cached polyline
thereafter. Over one scene: 270 paths planned, 2 replanned, **164,980 waypoint
queries served** — a 610× reduction in A* calls versus per-tick planning.

This matters for attribution: `shortest_pathfinder` rebuilds a whole
`PathFinderNew` (navmesh + BVH) on every call, and the A* underneath uses a
linear-scan open list. Caching helps *any* controller including the baseline, so
it is not a contribution of the behavior tree and must not be reported as one.

---

## 6. The behavior tree end to end

Induction in the navmesh frame (see 6.1 below), then the assembled tree driving
the runtime. Both fixes below were found by the crowd being visibly wrong, not
by a test.

| | Collision | raw | Density | Population | Kinematics | Travel Time | DTW | agents |
|---|---|---|---|---|---|---|---|---|
| CrowdES (20 trials) | 0.00781 | — | 0.0180 | 0.182 | 0.340 | 0.596 | 1.62 | 277 |
| waypoint + CBF@0.45 | 0.00259 | 0.00000 | 0.0195 | 0.199 | 0.451 | — | 2.23 | 264 |
| **BT + CBF@0.45** | **0.00249** | 0.00010 | 0.0196 | 0.200 | 0.467 | **0.576** | 2.08 | 282 |
| Ground truth | 0.0001 | — | — | — | — | — | — | 406 |

The tree reaches **3.1x fewer collisions than CrowdES** at comparable Density,
Population and Travel Time — it actually beats CrowdES on travel time — while
costing Kinematics (0.467 vs 0.340) and DTW (2.08 vs 1.62). Note it barely
improves on the *waypoint* floor, which is the comparison that matters: on this
evidence almost all of the collision reduction is the CBF, not the tree.

### 6.1 Induction must happen in the navmesh frame

Rotating futures so the waypoint lies along +x before clustering, as CrowdES
already does for its own endpoint clusters:

| | world frame | navmesh frame |
|---|---|---|
| ARI vs CrowdES B=8 | 0.448 | **0.544** |
| tree routing fidelity | 54.1% | **71.9%** |
| guards failing to separate | 1 of 7 (acc 0.574) | **0 of 7** (0.844–0.912) |
| max leaf dispersion | 0.579 m | 0.502 m |

Without it, three of eight leaves came out labelled `backstep` — they were the
agents walking toward world −x — and those primitives then dragged agents
travelling the other way backwards.

### 6.2 The DMP goal cannot be the navmesh waypoint

Design 05 sec 1.1 sets `g_l = c_{t,nav}`. That waypoint *recedes* as the agent
advances, so `(g - x)` never shrinks and the spring term never relaxes. With
`K = 100`, `tau = 1.8 s` and the measured median waypoint distance of 1.456 m,
the spring term alone commands **45 m/s², about 15x `a_max = 3`**. The primitive
saturates actuation permanently and the safety filter has no authority left:
minimum separation collapsed to 0.035 m against a `d_min` of 0.45, with the CBF
intervening on 41.6% of agent-ticks and agents drifting off their paths 42x more
often than the waypoint controller.

The fix is ordinary DMP semantics: each execution gets a **fixed** goal, set on
entry to the agent's position plus the leaf's own displacement rotated onto the
live navmesh direction. Interventions fell to 12.1%, replans from 84 to 5.

### 6.3 The action lifecycle is load-bearing

Design 05 sec 1.3 has an action leaf return SUCCESS at `||x - g|| <= eps`, after
which the tree re-ticks it and a fresh execution starts. Implementing leaf
*selection* without that lifecycle means an agent that keeps choosing the same
leaf reaches its goal once and then stops, because the spring has nothing left
to pull against:

| | selection only | + lifecycle |
|---|---|---|
| agents completing (GT 406) | 71 | **282** |
| Travel Time EMD | 11.895 | **0.576** |
| Population EMD | 1.169 | **0.200** |
| Kinematics | 3.368 | **0.467** |
| Density | 0.106 | **0.0196** |

## 7. Subsystem 7b: what could actually be checked

nuXmv and BehaVerify are external and licence-gated, so nothing depends on them.
`src/flow2bt/verification.py` emits the SMV model so the path stays open, and
`src/flow2bt/falsify.py` provides evidence that runs today.

### 7.1 The SMV model proves less than design 07 implies

The tree's control flow translates exactly — Sequence and Fallback are
deterministic given the guards, and each guard is a linear predicate — so the
tick is a boolean function of the feature vector. **The dynamics do not
translate.** The DMP integration, the CBF-QP and the agent model are absent, so
checking this file establishes properties of *which action is selected*, not of
the closed loop.

Design 07 sec 1.2 states its specs as
`LTLSPEC G !(env.distance_to_nearest_agent < 0.7)`. That is a closed-loop
property and is **not** established by model-checking the tree. The emitted file
carries that caveat as a comment, and the only `LTLSPEC` emitted is
`G (leaf >= 0)` — totality, which genuinely is about the tree.

### 7.2 Bounded falsification found the deadlock the design predicts

200 randomised episodes per configuration, driving the real controller, filter
and integrator (no scene, no emitter, so an episode costs milliseconds):

| scenario | `d_min` | worst separation | arrival rate | counterexamples |
|---|---|---|---|---|
| head-on | 0.20 | 0.1992 | 1.000 | none |
| head-on | 0.45 | 0.4498 | 0.840 | `F at_goal` in 8/50 |
| corridor | 0.20 | 0.2468 | 1.000 | none |
| corridor | 0.45 | 0.4507 | 1.000 | none |

**No collision counterexample anywhere.** The barrier converges to exactly
`d_min` — 0.4498 against 0.45 — which is the signature of it working, and which
is why `Episode.violations` carries a tolerance: without one, a filter doing its
job reports a violation every time on floating-point noise.

The `F at_goal` failures are a single family, and it is the one design 07
sec 1.2 anticipates. At **exactly zero** lateral offset the pairwise constraint
normal `-2*dp` lies along the approach axis, so the filter can brake but cannot
break the tie sideways. Both agents stop nose to nose on the barrier and neither
ever arrives:

| lateral offset | min separation | arrived | stalled |
|---|---|---|---|
| 0.00 m | 0.4499 | 0/2 | 2/2 |
| 0.01 m | 0.4499 | 2/2 | 0/2 |
| 0.20 m | 0.4499 | 2/2 | 0/2 |

One centimetre of asymmetry resolves it, and design 07 sec 1.2.3's proposed
remedy — an asymmetric tie-breaking guard — is the right shape of fix.

Note this is further evidence for a *small* `d_min`: at 0.2 m there are no
deadlocks at all, at 0.45 m they appear in 16% of head-on episodes.

### 7.3 A falsifier must not blame the controller for its own sampling

The first corridor sweep reported collisions in 29–39 of 60 episodes. They were
not the controller's: the generator drew lane positions independently, so agents
on the same side spawned millimetres apart — **128 of 200 episodes began inside
`d_min`**, worst gap 0.003 m — and a barrier cannot recover a state that starts
unsafe. Spacing the lanes helped; capping the agent count by what the corridor
physically holds (four abreast at 0.6 m needs 2.4 m, so eight agents in a 1.5 m
corridor is an infeasible request) fixed it: 4 of 300 episodes, worst gap
0.40 m. Only after that is the "no collision counterexamples" result above
meaningful.

## 8. Still open

* Scene evaluation with the finite stop leaf (§9.6), and the slow-leaf loop it
  leaves behind. The §6 row predates the fix, is a **single trial**
  against CrowdES's 20, and depended on the bug.
* The `d_min` frontier sweep across {0.2, 0.3, 0.45, 0.6}.
* `BTController` end-to-end in the runtime (the plumbing is tested; the induced
  bundle has not yet been run through a full scene).
* Subsystem 7b (BehaVerify / nuXmv). Deliberately not gating anything: nuXmv is
  external and licence-gated. The planned substitute is a bounded falsification
  harness against the real runtime.

## 9. Flow teacher -> tree: what the distillation costs

Teacher: `outputs/flow_eth/20260920-072219/checkpoints/epoch063-minfde0.0529.ckpt`.
All inductions in the navmesh frame, B=8, seed 0, train split.

### 9.1 Induction source ablation (in-sample, train split)

`routing_ceiling` is new: with several rollouts per state, members of one state
share a feature vector but can fall in different leaves, so no tree can exceed
the per-state majority share.

| source | trajectories | routing fidelity | ceiling | ARI vs CrowdES B=8 | leaf dispersion (m) | pooled RMS | guard CV acc |
|---|---|---|---|---|---|---|---|
| ground truth | 4096 x 1 | 71.9% | 100% | 0.544 | 0.14–0.50 | 0.253 | 0.844–0.912 |
| flow | 4096 x 1 | 71.9% | 100% | 0.592 | 0.17–0.46 | 0.247 | 0.733–0.922 |
| flow | 4096 x 4 | 70.7% | 93.2% | 0.565 | 0.17–0.46 | 0.254 | 0.789–0.910 |
| flow | 1024 x 16 | 73.5% | 93.3% | 0.588 | 0.13–0.42 | 0.232 | 0.804–0.927 |

The teacher's rollouts pick the same leaf ~93% of the time from one state, so
the teacher's own multimodality is only a ~7% effect at B=8 over 2 s. Induction
from it agrees somewhat better with CrowdES's modes, and the in-sample numbers
are otherwise indistinguishable from ground truth.

### 9.2 Open-loop agent-level evaluation (test split, 29,275 windows)

`src/evaluate_bt_openloop.py`. Same windows as `evaluate_agent.py`; no CBF, no
reacting neighbours. Leaf fixed from the observed window, DMP integrated at the
runtime's 50 ms tick with its lifecycle. ADE / FDE in metres.

| predictor | GT tree | flow x1 | flow x4 | flow x16 |
|---|---|---|---|---|
| oracle prototype (minADE₈ / minFDE₈) | **0.264** / 0.446 | 0.286 / 0.477 | 0.276 / 0.465 | 0.288 / 0.484 |
| routed prototype | **0.352** / 0.609 | 0.362 / 0.626 | 0.358 / 0.621 | 0.364 / 0.630 |
| routed DMP (what the runtime executes) | **0.350** / 0.611 | 0.360 / 0.627 | 0.356 / 0.622 | 0.361 / 0.631 |
| oracle DMP | **0.261** / 0.441 | 0.283 / 0.475 | 0.273 / 0.462 | 0.285 / 0.482 |
| held-out routing accuracy (nearest prototype) | 66.8% | 69.4% | 70.4% | 72.3% |

References on the same windows: flow teacher 0.2721 / 0.5321 (argmax latent),
minADE₂₀ 0.1562 / minFDE₂₀ 0.2727; CrowdES 0.2896 / 0.5659; **constant velocity
0.2159 / 0.4406**.

Read stage by stage:

* **Quantisation** (8 fixed shapes, perfect routing): 0.26–0.29 ADE. Already no
  better than the teacher's single argmax guess.
* **Routing** is the dominant loss: +0.07–0.09 m ADE, +0.15 m FDE. The tree ends
  up ~30% above the teacher on ADE and ~15% on FDE.
* **Execution** as a DMP costs nothing measurable once §9.4 is fixed.
* **The flow teacher does not buy a better tree here.** Flow-induced trees route
  a little more accurately but quantise worse, and the ground-truth tree has the
  lowest routed and oracle error.
* **Constant velocity beats every tree and the teacher** at this 2 s horizon on
  eth. Any open-loop claim for the tree has to be read against that.
* Held-out routing accuracy (66.8% for the GT tree) sits below the in-sample 71.9%.
  The labelling rule differs (nearest prototype in L2, not the dendrogram's
  embedding), so the two are only approximately comparable.

DMP replay error against each leaf's prototype is 0–7 cm at the 50 ms tick. The
induction report's `dmp_fit_residual` is the forcing-term regression residual in
forcing units, not a distance.

### 9.3 The runtime ticks at 50 ms, not 10 ms

`substeps: 4` at 5 fps gives `tick_dt = 0.05 s` (`flow2bt_framework.py:101`),
and the DMPs are stepped at that tick. The 10 ms figure is `rollout_dmp`'s default
integration step, which the runtime does not use.

### 9.4 Bug: the forcing term was applied on world axes

`BTController` rotated each leaf's *goal* onto the navmesh direction but handed
`DMPBank.acceleration` world-frame `goal - entry` for the per-dimension forcing
scale, with forcing weights fitted in the navmesh frame. A straight leaf has a
near-zero lateral `g - x0`, which the fit clamps to 1.0. Replayed along any
direction other than world +x, the along-track forcing landed on the wrong axis,
and the clamp inflated it up to ~100x.

Routed-DMP ADE before -> after the fix: GT tree 0.608 -> 0.350, flow x4
0.445 -> 0.356, flow x16 0.380 -> 0.361. `DMPBank.acceleration` now takes a
per-agent `rotation`, and `BTController` fixes one per execution alongside the
goal. `test_replay_is_rotation_equivariant` pins it. `runtime.rotate_forcing=false`
reproduces the old behaviour so the closed-loop numbers in §6 can be compared.

### 9.5 Closed loop after the fix: the stop leaf is absorbing

`seq_eth`, 5 trials, CBF@0.45, `rotate_forcing=true`:

| | Collision | raw | Density | Population | Kinematics | DTW | Travel Time | agents / trial |
|---|---|---|---|---|---|---|---|---|
| CrowdES (20 trials) | 0.00781 | — | 0.0180 | 0.182 | 0.340 | 1.62 | 0.596 | 277 |
| BT, GT tree, **pre-fix**, 1 trial (§6) | 0.00249 | 0.00010 | 0.0196 | 0.200 | 0.467 | 2.08 | 0.576 | 282 |
| BT, GT tree, fixed | 0.00664 ± 0.00834 | 0.0 | 0.0324 | 0.338 | 0.754 | 2.28 | 1.867 | 129 |
| BT, flow x16 tree, fixed | 0.00112 ± 0.00053 | 0.0 | 0.0513 | 0.549 | 1.487 | 2.58 | 4.616 | 188 |

Fixing §9.4 made every realism metric worse. The cause is structural, not the fix.
Every guard is dominated by the agent's own `speed`, and at speed ~0 every
guard on the path to the stop leaf passes. A stopped agent is therefore routed to
`stop`, which holds it at speed ~0, which routes it to `stop` again. Nothing in
the tree says "resume". Single agent, empty scene, waypoint 1.56 m ahead, 10 s:

| tree | start speed | progress, fixed | progress, pre-fix |
|---|---|---|---|
| GT | 1.3 m/s | 12.3 m (march) | 12.3–13.0 m |
| GT | 0 | **-0.08 m (stop, all headings)** | -0.08 m heading +x; **12.6 m / 2.7 m** heading +y / (-0.6,-0.8) |
| flow x16 | 0 | **0.03 m (stop, all headings)** | 0.03–0.34 m |

Before the fix, the mis-scaled forcing kicked stopped agents out of `stop` for
most headings: it was the resume mechanism, by accident. In a crowd the CBF brakes
agents to near zero, they latch into `stop`, and they never leave. So fewer
agents finish, density and population rise, and Travel Time blows up. The flow
tree latches harder: its stop leaf holds 38% of the induction set against 35%
for GT. **The §6 BT row, and Table 1's Flow2BT row, depended on this bug.**

A resume path needs a feature that the stop leaf does not drive to a fixed
point: time-in-leaf, preferred minus current speed, or a finite-duration `stop`
that returns SUCCESS and falls through.

### 9.6 Finite-duration stop leaf (implemented, not yet scene-evaluated)

`BTController(finite_stop=True)`, the default, and `runtime.finite_stop` in
`eval_flow2bt.yaml`:

* **Which leaves.** A leaf is *stationary* when its displacement is under
  `arrival_tolerance`. Such a leaf cannot complete by arrival: it "arrives" on
  the tick it starts. It is identified by geometry, not by name.
* **Duration.** The stationary leaf's `bt.Action` gets a terminator. It returns
  RUNNING for `stop_duration`, then FAILURE, and keeps returning FAILURE for
  `stop_refractory`. Both default to the leaf's `tau` (1.8 s).
* **Fall-through.** On FAILURE the guarded Sequence fails and the enclosing
  Fallback moves to its next child. In all four induced trees that child is a
  walking subtree. If nothing claims the agent, it is re-ticked with stop
  allowed, so the tree stays total.
* **Per-agent state is keyed by agent id.** The framework now passes
  `TickState.agent_ids`. Before, the controller dropped *every* agent's phase,
  goal and timers whenever the crowd changed size, which happens on most
  frames. A stop timer would rarely have expired.
* **Reset between trials.** `evaluate_flow2bt` resets the controller at the
  start of each trial, because agent ids restart per trial.

Same single-agent check as §9.5, starting at rest, 10 s:

| tree | finite_stop=false | finite_stop=true |
|---|---|---|
| GT | -0.08 m (stop) | 2.23 m: stop 1.8 s, then `swerve_right_2` at 0.52 m/s |
| flow x4 | -0.02 m | 1.59 m: stop, then `swerve_left_0` at 0.47 m/s |
| flow x16 | 0.03 m | 3.06 m: stop, then `march_2` at 0.38 m/s |

Identical across headings. The deadlock is gone, but a weaker version of the
same loop remains. The agent resumes into a *slow* leaf, the slow leaf keeps it
slow, and the speed-dominated guards keep choosing the slow leaf. It never gets
back to its 1.3 m/s preferred speed. Expect Travel Time to improve over §9.5 but
not to reach CrowdES. Closing this last loop needs a guard feature the leaf does
not control, such as preferred speed minus current speed.
