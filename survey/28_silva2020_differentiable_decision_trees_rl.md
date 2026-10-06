# Optimization Methods for Interpretable Differentiable Decision Trees in Reinforcement Learning

- **Authors:** Andrew Silva, Taylor Killian, Ivan D. Jimenez, Sung-Hyun Son, Matthew Gombolay
- **Year / Venue:** 2020 / Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics (AISTATS 2020), PMLR 108
- **Paper Link / Identifier:** [PMLR v108](https://proceedings.mlr.press/v108/silva20a.html) / [arXiv:1903.09338](https://arxiv.org/abs/1903.09338)
- **Code Repository:** [Official: CORE-Robotics-Lab/Interpretable_DDTS_AISTATS2020](https://github.com/CORE-Robotics-Lab/Interpretable_DDTS_AISTATS2020)
- **Primary Category:** Differentiable Optimization — gradient-based tree routing for RL (Pillar 1 / Differentiable BTs)

---

### 1. Executive Summary & Core Hypothesis

Decision trees cannot be updated online with SGD, so RL with tree policies lags deep RL. The authors hypothesize that soft oblique routing makes the full tree end-to-end differentiable, enabling policy-gradient / Q-learning updates over all nodes jointly, with a post-hoc crispification recovering an interpretable discrete tree without the batch-CART performance collapse.

---

### 2. Theoretical Framework & Mathematical Formulation

- **Input format:** Online state-action chains `(s, a, r, s')` from environment interaction; no teacher required (distillation optional).
- **Soft routing:** Each internal node `i` computes `p_i(x) = sigma(beta * (x^T w_i + b_i))` (right-branch probability) with inverse-temperature `beta`. Path probability to leaf `l`: `P^l(x) = prod p_i^{1[right]} (1-p_i)^{1[left]}`. Policy/value is the path-weighted leaf mixture.
- **Crispification:** After SGD, discretize each node to `1[w_i^T x + b_i >= 0]` (axis-aligned variant thresholds single features). Reports up to 7x average reward vs batch-trained CART after crispification.
- **Discrete structure extraction:** Fixed-depth full binary skeleton; learning discovers oblique split hyperplanes and leaf action/Q distributions. No grammar search (unlike Huang 2025) and no agglomerative clustering.
- **Optimization scheme:** Policy-gradient (`nabla J = E[sum_t G_t nabla log pi_T(a_t|s_t)]` through soft paths) and tree-Q variants with RMSProp/Adam; entropy regularization to keep branch mass alive; temperature schedule on `beta`.

### 3. Architecture & Neural Integration

- **Neural Role:** None required — the tree itself is the differentiable function approximator. Optionally warm-started from expert rules (ProLoNets lineage).
- **Interface / Boundary:** Gradient flows through routing sigmoids to split weights `w_i` and leaf logits jointly. Guards `C(s)` are oblique hyperplanes `w^T s + b >= 0` — the direct remedy for Flow2BT `src/flow2bt/conditions.py` univariate collapse (speed-dominated SVMs at 66.8-72.3% fidelity). Leaves hold discrete actions or Q-vectors.

### 4. Empirical Evaluation & Benchmarks

- **Environments / Tasks:** Classic control (CartPole, MountainCar, LunarLander) plus discrete-action Gym suites; human interpretability user study (tree vs rule list vs MLP).
- **Baseline Comparisons:** MLPs (parity or better claimed on all domains); batch-trained CART; rule lists; ablations on depth, `beta`, and crispification gap.
- **Key Quantitative Results:** Equals or beats MLP return on tested domains; online DDT up to 7x batch-tree reward; user study `p < 0.001` favoring trees for simulatability.

### 5. Failure Modes, Trade-offs & Limitations

- **Interpretability Preservation:** Oblique splits (`sum w_j s_j`) are less simulatable than axis-aligned ones; crispification can drop return if soft mass was shared across contradictory leaves (same discretization gap as Huang 2025).
- **Scalability & Gradient Stability:** Path products `prod p_i` vanish in deep trees (`D > ~8`); needs careful `beta` annealing. Fixed skeleton cannot grow/prune topology.
- **Unaddressed Gaps:** Single-pass DT inference; no BT Sequence/Fallback ticking, no `RUNNING` persistence, no safety filter. Delayed bifurcations learnable only if reward gradient reaches early splits — no explicit latent intent.

---

### 6. Relevance & Critical Assessment for Flow2BT Distillation

- **Resolution of the Decodability Gap:** Soft routing keeps counterfactual branch mass differentiable, so `s_0` guards receive gradient from late-bifurcation rewards — a gradient analogue to VIPER's Q-weighting. Still reactive (no memory); combine with LEAPS-style latent `z` for persistent intent.
- **Decision Trees vs. Behavior Trees Gap:** Same compilation story as VIPER: oblique DT -> BT via nested Fallbacks. Tick semantics must be added externally.
- **Compatibility with Flow2BT Subsystems:** Most actionable for Subsystem 4: replace linear SVM guards with oblique DDT nodes trained by policy gradient on flow-teacher rollouts, optimizing multi-feature predicates `C_k(s) = 1[sum w_j s_j + b >= 0]` over `(TTC, d_lat, v_rel)` instead of speed-collapsed univariate splits. Pair with DisCo-DSO when decoupled discrete feature selection is preferred over joint gradient search.
