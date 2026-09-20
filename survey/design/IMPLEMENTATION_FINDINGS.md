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

* Flow teacher (Subsystem 2) is training; induction from `source=flow` is the
  design's actual proposal and has not yet been compared against
  `source=ground_truth`.
* The `d_min` frontier sweep across {0.2, 0.3, 0.45, 0.6}.
* `BTController` end-to-end in the runtime (the plumbing is tested; the induced
  bundle has not yet been run through a full scene).
* Subsystem 7b (BehaVerify / nuXmv). Deliberately not gating anything: nuXmv is
  external and licence-gated. The planned substitute is a bounded falsification
  harness against the real runtime.
