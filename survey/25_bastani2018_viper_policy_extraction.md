# Verifiable Reinforcement Learning via Policy Extraction (VIPER)

- **Authors:** Osbert Bastani, Yewen Pu, Armando Solar-Lezama
- **Year / Venue:** 2018 / Advances in Neural Information Processing Systems 31 (NeurIPS 2018)
- **Paper Link / Identifier:** [NeurIPS 2018](https://proceedings.neurips.cc/paper_files/paper/2018/hash/e6d8545daa42d5ced125a4bf747b3688-Abstract.html) / [arXiv:1805.08328](https://arxiv.org/abs/1805.08328)
- **Code Repository:** [Official: obastani/viper](https://github.com/obastani/viper) (re-implementation: [Safe-RL-Team/viper-verifiable-rl-impl](https://github.com/Safe-RL-Team/viper-verifiable-rl-impl))
- **Primary Category:** Generative & Distillation Approaches — oracle-guided decision-tree policy extraction (Pillar 3 / Policy Distillation)

---

### 1. Executive Summary & Core Hypothesis

Deep RL oracles achieve high return but are unverifiable. Decision-tree (DT) policies are nonparametric (expressive) yet structured (verifiable via SMT solvers such as Z3/Reluplex). The bottleneck is that DT policies are hard to train directly, even on CartPole. VIPER hypothesizes that imitation of a DNN oracle plus its Q-function, with DAGGER aggregation and Q-loss weighting, suffices to distill a small CART tree matching oracle return while enabling proofs of robustness, non-loss, and stability.

---

### 2. Theoretical Framework & Mathematical Formulation

- **Input format:** State-action demonstration rollouts sampled from a pretrained DNN oracle policy `pi_star(s)` and its Q-function `Q_star(s,a)`. No unlabelled `tau` alone: labels come from the teacher.
- **Resampling / critical-state weighting:** Define the imitation gap `l(s,a) = V_star(s) - Q_star(s,a)` where `V_star(s) = max_a Q_star(s,a)`. States with large gap (wrong action is costly) are oversampled: `p(s) propto l(s, pi_T(s))`.
- **DAGGER + CART objective:** At iteration `i`, aggregate `D <- D U {(s, pi_star(s)) : s ~ d_{pi_hat}}` under the current student visitation, then fit:
  `pi_T = argmin_{T in DT} sum_{(s,a_star) in D} l(s, pi_T(s)) * 1[pi_T(s) != a_star]`
  implemented as sample-weighted CART (`sklearn.tree.DecisionTreeClassifier` with `sample_weight=l(s)`).
- **Discrete structure extraction:** Greedy top-down impurity minimization on the Q-weighted loss. No Gumbel-Softmax, no VAE; discreteness is native. Depth is capped (typically `<= 10`) and the best-of-N trees on validation return is kept.
- **Optimization scheme:** Two-phase: (1) DRL trains oracle (e.g. PPO/DQN); (2) imitation loop above. No end-to-end gradient through the tree.

### 3. Architecture & Neural Integration

- **Neural Role:** DNN oracle + Q-network only at train time. At test time the policy is a pure axis-aligned tree.
- **Interface / Boundary:** Teacher labels and Q-gaps interface with CART sample weights. Student is standalone C++/Python `if s_j <= theta` cascade.

### 4. Empirical Evaluation & Benchmarks

- **Environments / Tasks:**
  - Toy Pong variant (provably never loses).
  - Atari Pong with symbolic state (provable robustness to bounded pixel/state perturbation).
  - CartPole (Lyapunov-based provable stability region).
- **Baseline Comparisons:**
  - Naive behavioral cloning / DAGGER without Q-weighting.
  - Q-DAGGER ablation.
  - Oracle DNN (parity target).
  - Tree-based batch RL (Ernst et al. 2005) — shown not to scale to CartPole.
- **Key Quantitative Results:**
  - Matches oracle reward with depth `<= 10` trees on all three domains.
  - Enables Z3/Reluplex proofs (robustness window, non-losing strategy, region of attraction) intractable for the DNN.

### 5. Failure Modes, Trade-offs & Limitations

- **Interpretability Preservation:** High — axis-aligned predicates are auditable, but deep trees (>15 nodes) still require proof assistants to trust.
- **Scalability & Gradient Stability:** CART greedy splitting is myopic; Q-weighting mitigates but does not guarantee global optimality. No gradient signal; high-dimensional visual inputs require symbolic abstraction first.
- **Unaddressed Gaps:** Single-pass DT only: no BT ticks, no `{SUCCESS, FAILURE, RUNNING}`, no Fallback preemption. No memory: aliased `s_0` states that diverge later must be separated by feature engineering.

---

### 6. Relevance & Critical Assessment for Flow2BT Distillation

- **Resolution of the Decodability Gap:** DAGGER aggregation explicitly revisits states the student (not teacher) visits, and Q-weighting forces an early split at `t=0` when a late bifurcation is costly — the closest imitation-learning analogue to solving state aliasing without memory.
- **Decision Trees vs. Behavior Trees Gap:** Output is a flat DT. Compilation to BT is mechanical (`if -> Sequence`, `else -> Fallback`) but adds no reactivity beyond the 1-step map unless combined with `06_reactive_bt_assembly_execution.md` ticking.
- **Compatibility with Flow2BT Subsystems:** Drop-in teacher-student distiller for Subsystem 4 (guard induction) and Subsystem 5 (leaf compression) when the teacher is a flow/diffusion policy: replace `pi_star = argmax Q_star` with `pi_star = flow-mode sampler` and `l(s)` with trajectory-cost gap. Does not exploit vector-field geometry `v_theta(x,t)` (cf. Subsystem 3); pair with reverse-time clustering for topology, VIPER for guard thresholds.
