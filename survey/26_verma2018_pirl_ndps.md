# Programmatically Interpretable Reinforcement Learning (PIRL / NDPS)

- **Authors:** Abhinav Verma, Vijayaraghavan Murali, Rishabh Singh, Pushmeet Kohli, Swarat Chaudhuri
- **Year / Venue:** 2018 / Proceedings of the 35th International Conference on Machine Learning (ICML 2018), PMLR 80:5045-5054
- **Paper Link / Identifier:** [PMLR v80](http://proceedings.mlr.press/v80/verma18a.html) / [arXiv:1804.02477](https://arxiv.org/abs/1804.02477) / [PDF](http://proceedings.mlr.press/v80/verma18a/verma18a.pdf)
- **Code Repository:** No official repo indexed; unofficial mirrors: [VAIBHAV-2303/PiRL](https://github.com/VAIBHAV-2303/PiRL), [andresokol/gandrl_nps](https://github.com/andresokol/gandrl_nps) — treat as partially reproducible
- **Primary Category:** Generative & Distillation Approaches — neurally directed program search (Pillar 3 / Policy Distillation)

---

### 1. Executive Summary & Core Hypothesis

Neural policies are performant but uninterpretable and unverifiable. Programmatic policies in a domain-specific language (DSL) are auditable and amenable to symbolic verification. Direct search over programs for maximal reward is a nonsmooth combinatorial problem. PIRL hypothesizes that a neural oracle can direct that search: first train a DRL policy, then perform local sketch-based search for the program closest to the oracle, inheriting oracle performance with program structure.

---

### 2. Theoretical Framework & Mathematical Formulation

- **Input format:** Demonstration rollouts `(s, a_star=pi_oracle(s))` from a DRL oracle in TORCS, plus environment reward for final selection.
- **Neurally Directed Program Search (NDPS):** Given DSL `E` (conditionals over sensor detectors + PID/steering primitives with free parameters `theta`):
  1. Train oracle `pi_theta` via DDPG/DRL.
  2. Synthesize `e_hat = argmin_{e in E} d(e, pi_oracle) + Reg(e)` where `d(e, pi) = E_{s ~ d_e}[||e(s) - pi(s)||]` is trajectory distance under the program's own visitation (DAGGER-style resampling), and `Reg` penalizes program length.
  3. Sketch enumeration: fix control-flow skeleton, optimize continuous parameters via Bayesian/global optimization; iterate skeletons by local rewrite rules.
- **Discrete structure extraction:** Enumerative syntax-guided synthesis over the DSL grammar, not Gumbel relaxation or gradient routing. Discreteness preserved end-to-end.
- **Optimization scheme:** DRL (policy gradient) for oracle; combinatorial local search + continuous parameter tuning for program. Selection by validation lap time / reward.

### 3. Architecture & Neural Integration

- **Neural Role:** Oracle network used only to generate labels and guide neighborhood choice during search. Final artifact contains no network.
- **Interface / Boundary:** Oracle distance `d(e, pi)` is the sole neural-symbolic interface. Guards `C(s)` are DSL predicates over hand-designed detectors (track position, speed, lidar); leaves `A(s)` are PID/steering/throttle primitives.

### 4. Empirical Evaluation & Benchmarks

- **Environments / Tasks:** TORCS car-racing simulator (multiple tracks, unseen-track transfer).
- **Baseline Comparisons:** DDPG/DRL oracle; hand-coded controllers.
- **Key Quantitative Results:** Discovered human-readable policies clearing performance bars (full laps at competitive speed); smoother steering trajectories than DRL oracle; better zero-shot transfer to tracks not seen in training.

### 5. Failure Modes, Trade-offs & Limitations

- **Interpretability Preservation:** High — programs read as nested `if` driving rules. But interpretability depends on DSL designer; opaque detectors leak neural complexity back in.
- **Scalability & Gradient Stability:** Sketch enumeration scales exponentially in program length; limited to tens of lines. No gradient through structure; search needs many oracle queries.
- **Unaddressed Gaps:** DSL is not a BT (no tick, no `{SUCCESS, FAILURE, RUNNING}`). No generative temporal prior (flow/diffusion); no latent branching variable for delayed bifurcations.

---

### 6. Relevance & Critical Assessment for Flow2BT Distillation

- **Resolution of the Decodability Gap:** Oracle-guided resampling biases search toward programs that commit early to the correct branch when late divergence is expensive, yielding smoother preemptive guards. No explicit belief/memory state.
- **Decision Trees vs. Behavior Trees Gap:** DSL `if/else` chains compile directly to BT `Fallback` + `Sequence` (condition -> action), making PIRL programs the closest pre-BT programmatic analogue in this set after Huang 2025.
- **Compatibility with Flow2BT Subsystems:** Template for Subsystem 6 (reactive assembly): replace TORCS oracle with flow teacher `v_theta(x,t)` rollouts and DSL with BT grammar; NDPS becomes flow-directed BT sketch search. Complements Subsystem 3 clustering (topology proposals) with a reward-grounded selection rule.
