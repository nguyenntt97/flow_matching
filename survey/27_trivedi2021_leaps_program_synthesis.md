# Learning to Synthesize Programs as Interpretable and Generalizable Policies (LEAPS)

- **Authors:** Dweep Trivedi, Jesse Zhang, Shao-Hua Sun, Joseph J. Lim
- **Year / Venue:** 2021 / Advances in Neural Information Processing Systems 34 (NeurIPS 2021)
- **Paper Link / Identifier:** [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/d37124c4c79f357cb02c655671a432fa-Abstract.html) / [arXiv:2108.13643](https://arxiv.org/abs/2108.13643)
- **Code Repository:** [Official: clvrai/leaps](https://github.com/clvrai/leaps) / [Project site](https://clvrai.com/leaps)
- **Primary Category:** Generative Approaches — latent program embedding + search (Pillar 3 / Generative Policy Design)

---

### 1. Executive Summary & Core Hypothesis

Searching program space from scratch with reward only is intractable. LEAPS hypothesizes that diverse program behaviors lie on a smooth latent manifold learnable unsupervised from random programs, and that reward-driven search in that continuous space (rather than discrete syntax space) reliably yields interpretable, generalizable policies without demonstrations or I/O specs.

---

### 2. Theoretical Framework & Mathematical Formulation

- **Input format:** Reward-only trajectories `tau = (s_0, ..., s_T)`; no demonstrations. Stage 1 uses a synthetic corpus of random Karel programs and their execution traces.
- **Stage 1 — Program embedding VAE:** Encoder `q_phi(z | prog)` / decoder `p_theta(prog | z)` over tokenized Karel DSL, jointly trained with:
  1. Program reconstruction `E[log p_theta(prog | z)]`,
  2. Program behavior reconstruction (predict execution trace / return from `z`),
  3. Latent behavior reconstruction (align trace embedding with `z`), plus `beta * KL(q || N(0,I))`.
- **Stage 2 — Latent search:** Freeze decoder; run Cross-Entropy Method (CEM) over `z` to maximize `E[R(Decode(z))]`: sample population `{z_i}`, evaluate episodic return, refit elite Gaussian, repeat. Best `Decode(z_star)` is the policy.
- **Discrete structure extraction:** VAE decoder maps continuous `z` to discrete Karel tokens (`if/while/repeat`, `move/turn/pick/put`, predicates). Syntax mask enforces validity. No Gumbel annealing at search time.
- **Optimization scheme:** Supervised VAE training (SGD) followed by derivative-free CEM on returns. Risk-seeking only implicitly via elite refitting.

### 3. Architecture & Neural Integration

- **Neural Role:** GRU/Transformer encoder-decoder parameterizes the program manifold; a behavior encoder grounds `z` in execution semantics. No network at deployment — pure Karel program.
- **Interface / Boundary:** Latent vector `z` is the neural-symbolic interface. Guards `C(s)` are Karel predicates (`frontIsClear`, `markerPresent`); leaves `A(s)` are grid actions. Control flow (`if/while`) maps to BT `Fallback/Sequence` with loop decorators.

### 4. Empirical Evaluation & Benchmarks

- **Environments / Tasks:** Karel gridworld suite (TopOff, Seeder, Snake, DoorKey, StairClimber variants); generalization to larger maps; MuJoCo discussion.
- **Baseline Comparisons:** PPO/DRL; program synthesis baselines (EC, DreamCoder-style, VIPER-style distillation); ablations removing each of the three embedding losses.
- **Key Quantitative Results:** Reliably synthesizes solving programs where from-scratch search fails; outperforms DRL and synthesis baselines on solve rate; retains performance on larger unseen maps; human debugging study favors programs over neural policies.

### 5. Failure Modes, Trade-offs & Limitations

- **Interpretability Preservation:** High — Karel programs are short and human-debuggable. But latent space itself is opaque; two nearby `z` can decode to syntactically distant programs.
- **Scalability & Gradient Stability:** VAE needs a large random-program corpus; CEM needs hundreds of rollout batches per task. Karel DSL is discrete-grid; continuous robotics guards need re-grounding.
- **Unaddressed Gaps:** No BT tri-state (`RUNNING`); synchronous grid steps, no reactive tick preemption. No flow/diffusion prior — corpus is random programs, not trajectory ensembles.

---

### 6. Relevance & Critical Assessment for Flow2BT Distillation

- **Resolution of the Decodability Gap:** Strongest match in this set: latent `z` persists intent across the full `tau`, so a program decoded from `z` can branch on `s_0`-observable predicates that anticipate late bifurcations — exactly the memory lift the State-Decodability Bottleneck calls for. Generalization to larger maps evidences early-commit guards.
- **Decision Trees vs. Behavior Trees Gap:** Karel control flow is hierarchical and translatable to BTs, but lacks tick semantics and CBF safety wrapping (cf. `07_safety_cbf_formal_verification.md`).
- **Compatibility with Flow2BT Subsystems:** Blueprint for a flow-conditioned variant: replace random-program corpus with flow-teacher trajectory modes (Subsystem 2) to learn a behavior manifold, then CEM-search for the BT whose leaves (Subsystem 5 DMPs) cover those modes. Pairs naturally with DisCo-DSO for guard threshold tuning (Subsystem 4).
