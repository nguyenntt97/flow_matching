# Trees to Flows and Back: Unifying Decision Trees and Diffusion Models

- **Authors:** Sai Niranjan Ramachandran, Suvrit Sra
- **Year / Venue:** 2026 / arXiv preprint (arXiv:2605.00414 [cs.LG, cs.AI, cond-mat.stat-mech])
- **Paper Link / Identifier:** [arXiv:2605.00414](https://arxiv.org/abs/2605.00414) / [https://doi.org/10.48550/arXiv.2605.00414](https://doi.org/10.48550/arXiv.2605.00414)
- **Primary Category:** Differentiable BT

### 1. Executive Summary & Core Hypothesis
This foundational paper establishes a crisp, formal mathematical duality between discrete hierarchical decision trees and continuous-time diffusion processes (stochastic differential equations and probability flow ODEs). The authors hypothesize and theoretically prove that the hierarchical spatial partitioning of a decision tree corresponds to a coarse-to-fine entropy-reducing Markov chain whose infinitesimal continuum limit converges to a drift-diffusion process governed by the Fokker-Planck equation. Grounded in this equivalence, the paper introduces a unified optimization objective—Global Trajectory Score Matching (GTSM)—showing that functional gradient tree boosting is asymptotically optimal for trajectory score matching, and provides two practical algorithms: TreeFlow (tree-conditioned continuous flow matching) and DSM-Tree (distilling hierarchical decision logic into continuous neural representations).

### 2. Theoretical Framework & Mathematical Formulation
- **Node Mechanics & Return Status Handling:** In classical decision trees, a sample $\mathbf{x} \in \mathcal{X}$ traverses routing predicates $h_n(\mathbf{x}) = \mathbb{I}(\mathbf{x}_{j} \le \theta)$ deterministically. The authors reformulate tree traversal as a discrete-time Markov chain of nested spatial partitions $\{\Pi_k\}_{k=0}^T$, where $k=0$ corresponds to the fine-grained leaf partition $\Pi_0 = \{R_1, \dots, R_M\}$ and $k=T$ corresponds to the trivial root partition $\Pi_T = \{\mathcal{X}\}$. Each partition $\Pi_k$ induces a piecewise constant probability density:
  $$p(\mathbf{x}, k) = \sum_{R \in \Pi_k} \frac{P(R)}{\text{Vol}(R)} \mathbb{I}(\mathbf{x} \in R)$$
  The transition operator $\mathcal{M}_k: p(\mathbf{x}, k-1) \mapsto p(\mathbf{x}, k)$ is defined as the conditional expectation with respect to the sub-$\sigma$-algebra $\mathcal{F}_k = \sigma(\Pi_k)$, representing an irreversible, entropy-increasing coarse-graining step satisfying probability conservation:
  $$\int_A p(\mathbf{x}, k) d\mathbf{x} = \int_A p(\mathbf{x}, k-1) d\mathbf{x}, \quad \forall A \in \mathcal{F}_k$$
  In the continuous-time limit, the discrete status is relaxed into a continuous probability density $p(\mathbf{x}, t)$ governed by a continuous vector field.
- **Execution / Control Flow:** The authors construct a dyadic refinement sequence $\{\mathcal{T}^{(n)}\}_{n=0}^\infty$ inserting intermediate partitions with complexity decay $\mathcal{O}(2^{-n})$. By expanding the transition propagator via the Kramers-Moyal expansion:
  $$\frac{\partial p(\mathbf{x}, t)}{\partial t} = \sum_{m=1}^\infty \frac{(-1)^m}{m!} \sum_{i_1,\dots,i_m} \frac{\partial^m}{\partial x_{i_1}\dots\partial x_{i_m}} \left[ \alpha_{i_1\dots i_m}^{(m)}(\mathbf{x}, t) p(\mathbf{x}, t) \right]$$
  and establishing sample path continuity ($\lim_{\Delta t \to 0} \frac{1}{\Delta t} \int_{\|\mathbf{y}-\mathbf{x}\|>\epsilon} P(\mathbf{y}, t+\Delta t \mid \mathbf{x}, t) d\mathbf{y} = 0$), Pawula's Theorem truncates the expansion at $m=2$. This yields the continuous-time Fokker-Planck equation:
  $$\frac{\partial p(\mathbf{x}, t)}{\partial t} = -\nabla \cdot (\mu(\mathbf{x}, t) p(\mathbf{x}, t)) + \frac{1}{2} \nabla \cdot \nabla \cdot (D(\mathbf{x}, t) p(\mathbf{x}, t))$$
  with equivalent Itô SDE $d\mathbf{x}_t = \mu(\mathbf{x}_t, t) dt + \sigma(t) d\mathbf{w}_t$ and deterministic probability flow ODE:
  $$\frac{d\mathbf{x}}{dt} = \mu(\mathbf{x}, t) - \frac{1}{2} \sigma^2(t) \nabla_\mathbf{x} \log p_t(\mathbf{x})$$
- **Optimization Strategy:** The authors introduce the Global Trajectory Score Matching (GTSM) objective across the continuum of intermediate distributions:
  $$\mathcal{L}_{\text{GTSM}}(\theta) = \mathbb{E}_{t \sim [0, T]} \left[ \lambda(t) \mathbb{E}_{\mathbf{x} \sim p_t} \left[ \| s_\theta(\mathbf{x}, t) - \nabla_\mathbf{x} \log p_t(\mathbf{x}) \|_2^2 \right] \right]$$
  They prove via Bellman optimality that greedy functional gradient boosting updates $F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \gamma_m h_m(\mathbf{x})$ minimize the discrete-time residual:
  $$r_{i, m} = -\left[ \frac{\partial L(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)} \right]_{F=F_{m-1}}$$
  which is an unbiased Monte Carlo estimator of the integrated score matching error $\int_{t_m}^{t_{m+1}} \| \nabla_\mathbf{x} \log p_t^\star(\mathbf{x}) - \nabla_\mathbf{x} \log p_t^{(m)}(\mathbf{x}) \|^2 dt$.

### 3. Architecture & Neural Integration
- **Neural Role:** The framework establishes a bidirectional neural-symbolic bridge:
  1. *DSM-Tree (Distilled Score Matching Tree):* A multi-task neural network $f_\theta(\mathbf{x}, t)$ acts as a student model learning the continuous hierarchical score fields generated by an ensemble of decision trees, matching teacher decision boundaries across all partition scales simultaneously.
  2. *TreeFlow (Tree-Conditioned Continuous Normalizing Flow):* A continuous neural vector field $v_\theta(\mathbf{x}, t)$ parameterized by deep residual networks or MLPs is conditioned on hierarchical tree paths $\mathbf{c} = \mathcal{T}_{\text{path}}(\mathbf{x})$, enabling targeted generative flow matching along tree-structured partition trajectories.
- **Interface / Boundary:** Continuous neural features interface with discrete tree logic via hierarchical conditioning vectors: tree routing paths are encoded as discrete token sequences or leaf embedding matrices that modulate the hidden layers of continuous neural ordinary differential equation (neural ODE) drift functions through adaptive layer normalization (AdaLN).

### 4. Empirical Evaluation & Benchmarks
- **Environments / Tasks:** 
  - Synthetic multimodal distributions (Nested Swiss Roll, 8-Gaussian Mixture, Concentric Rings) to verify theoretical dendrogram extraction and score field convergence.
  - Tabular discriminative benchmarks from UCI Machine Learning Repository and OpenML (Adult Census, Covertype, California Housing, Higgs Boson).
  - Tabular generative modeling benchmarks evaluating density estimation, mutual information preservation, and synthetic sample fidelity.
- **Baseline Comparisons:** 
  - Standard gradient boosted decision trees: XGBoost, LightGBM, CatBoost.
  - Generative tabular baselines: TabDDPM, CTGAN, TVAE, conditional Flow Matching (CFM).
  - Neural tabular models: FT-Transformer, SAINT, TabNet.
- **Key Quantitative Results:**
  - *Generative Fidelity & Speed:* TreeFlow achieved a $2\times$ reduction in sampling steps compared to TabDDPM while achieving superior Wasserstein-1 distance and coverage metrics across tabular distributions ($p < 0.01$).
  - *Neural Distillation:* DSM-Tree distilled 500-tree XGBoost ensembles into a compact 4-layer MLP student matching teacher AUROC within $1.4\%$ on Covertype and $0.8\%$ on Adult Census, drastically compressing inference memory footprints.

### 5. Failure Modes, Trade-offs & Limitations
- **Interpretability Preservation:** The continuous-time flow relaxation smooths the axis-aligned orthogonal boundaries of decision trees into smooth, curved vector fields. While the global hierarchical topology is preserved via dendrogram extraction from the continuous flow trajectories, the direct step-by-step human readability of boolean predicates is transferred into continuous score gradients $\nabla_\mathbf{x} \log p_t(\mathbf{x})$.
- **Scalability & Gradient Stability:** Evaluating high-dimensional drift-diffusion equations incurs quadratic Jacobian trace computation costs ($\mathcal{O}(D^2)$) during continuous log-likelihood evaluation, requiring Hutchinson stochastic trace estimation. Extreme density discontinuities at fine tree leaf boundaries can introduce localized gradient stiffening during ODE integration.
- **Unaddressed Gaps:** The formulation assumes static, non-reactive input distributions. It does not natively account for closed-loop reactive control semantics (such as Behavior Tree execution ticks, state transitions driven by external agent actions, or non-deterministic environment feedback).

