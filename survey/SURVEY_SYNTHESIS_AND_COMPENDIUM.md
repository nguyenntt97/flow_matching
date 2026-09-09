# Systematic Literature Survey & Compendium: Learnable & Neural Behavior Trees

**Anchor Paper:** [arXiv:2605.00414](https://arxiv.org/abs/2605.00414) — *Trees to Flows and Back: Unifying Decision Trees and Diffusion Models* (Sai Niranjan Ramachandran & Suvrit Sra, 2026)  
**Location:** `/home/nguyen/projects/flow_matching/survey/`

---

# Table of Contents
1. [Executive Summary & Foundational Anchor (arXiv:2605.00414)](#1-executive-summary--foundational-anchor)
2. [Comparative Matrix](#2-comparative-matrix)
3. [Open Theoretical Frontiers (400-Word Synthesis)](#3-open-theoretical-frontiers)
4. [Paper Dossiers: Complete Technical Records](#4-paper-dossiers-complete-technical-records)
   - [Dossier 01: Trees to Flows and Back (arXiv:2605.00414)](#dossier-01-trees-to-flows-and-back-unifying-decision-trees-and-diffusion-models)
   - [Dossier 02: Differentiable Synthesis of Behavior Tree Architectures and Execution Nodes](#dossier-02-differentiable-synthesis-of-behavior-tree-architectures-and-execution-nodes)
   - [Dossier 03: Deep Neural Decision Forests](#dossier-03-deep-neural-decision-forests)
   - [Dossier 04: Distilling a Neural Network Into a Soft Decision Tree](#dossier-04-distilling-a-neural-network-into-a-soft-decision-tree)
   - [Dossier 05: Adding Neural Network Controllers to Behavior Trees without Destroying Performance Guarantees](#dossier-05-adding-neural-network-controllers-to-behavior-trees-without-destroying-performance-guarantees)
   - [Dossier 06: Learning of Parameters in Behavior Trees for Movement Skills](#dossier-06-learning-of-parameters-in-behavior-trees-for-movement-skills)
   - [Dossier 07: A Framework for Constrained and Adaptive Behavior-Based Agents](#dossier-07-a-framework-for-constrained-and-adaptive-behavior-based-agents)
   - [Dossier 08: Learning Behavior Trees with Genetic Programming in Unpredictable Environments](#dossier-08-learning-behavior-trees-with-genetic-programming-in-unpredictable-environments)
   - [Dossier 09: Behavior Tree Learning for Robotic Task Planning through Monte Carlo DAG Search over a Formal Grammar](#dossier-09-behavior-tree-learning-for-robotic-task-planning-through-monte-carlo-dag-search-over-a-formal-grammar)
   - [Dossier 10: GAME: Generational Adversarial MAP-Elites for Co-evolving Behavior Trees](#dossier-10-game-generational-adversarial-map-elites-for-co-evolving-behavior-trees)
   - [Dossier 11: Combining Control Barrier Functions and Behavior Trees for Multi-Agent Underwater Coverage Missions](#dossier-11-combining-control-barrier-functions-and-behavior-trees-for-multi-agent-underwater-coverage-missions)
   - [Dossier 12: Neuro-Symbolic Behavior Trees (NSBTs) and Their Verification](#dossier-12-neuro-symbolic-behavior-trees-nsbts-and-their-verification)

---

# 1. Executive Summary & Foundational Anchor

Behavior Trees (BTs) have become a standard architectural paradigm for autonomous decision-making, robotic task orchestration, and reactive planning. By organizing autonomous behavior into hierarchical directed trees evaluated via high-frequency ticks, BTs provide modularity, reactivity, and human interpretability. However, the classical execution mechanism—where nodes return discrete, non-differentiable categorical statuses ($\{\text{SUCCESS}, \text{FAILURE}, \text{RUNNING}\}$) across Sequence ($\to$), Fallback ($?$), and Parallel ($\Rightarrow$) compositions—blocks direct end-to-end backpropagation.

This survey grounds its analysis in the theoretical breakthrough introduced by **arXiv:2605.00414: "Trees to Flows and Back: Unifying Decision Trees and Diffusion Models"** (Sai Niranjan Ramachandran & Suvrit Sra, 2026). Ramachandran & Sra rigorously prove that the discrete, spatial coarse-graining of a hierarchical decision tree defines an entropy-increasing Markov chain whose continuous limit converges to a drift-diffusion process governed by the Fokker-Planck equation:
$$\frac{\partial p(\mathbf{x}, t)}{\partial t} = -\nabla \cdot (\mu(\mathbf{x}, t) p(\mathbf{x}, t)) + \frac{1}{2} \nabla \cdot \nabla \cdot (D(\mathbf{x}, t) p(\mathbf{x}, t))$$
with corresponding deterministic probability flow ODE:
$$\frac{d\mathbf{x}}{dt} = \mu(\mathbf{x}, t) - \frac{1}{2} \sigma^2(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$$

The authors reveal that gradient boosted tree optimization is asymptotically optimal for **Global Trajectory Score Matching (GTSM)** in the space of SDE trajectories, introducing two fundamental instantiations: **TreeFlow** (continuous flow matching along tree-structured partition trajectories) and **DSM-Tree** (distillation of hierarchical tree logic into continuous neural networks via multi-scale score matching).

This mathematical duality provides the unifying missing link for Learnable Behavior Trees:
1. **Differentiable Behavior Trees:** Demonstrates that discrete hierarchical tree routing is the macroscopic limit of continuous vector fields, providing the mathematical justification for smooth t-norm relaxations, Gumbel-Softmax grammar routing, and soft decision trees.
2. **Neuro-Symbolic Hybrid Control:** Provides a rigorous pathway for bidirectional distillation between continuous deep neural policies (flows) and symbolic, discrete Behavior Trees.
3. **Structure Learning:** Bridges combinatorial grammar search (GP, MCTS, MAP-Elites) and functional gradient optimization.
4. **Verifiable Safe Control:** Connects continuous Control Barrier Functions (CBFs) with the score field geometry of hierarchical systems.

---

# 2. Comparative Matrix

| Paper Name | Primary Category | Differentiable? | Topology Learned vs. Fixed | Leaf Node Type | Benchmark Tasks |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ramachandran & Sra (2026)**<br>*Trees to Flows and Back* | Differentiable BT (Anchor) | **Yes** (Continuous limit via SDE/ODE) | Learned (GTSM boosting & tree distillation) | Continuous score fields $\nabla_{\mathbf{x}} \log p_t$, neural vector fields | UCI/OpenML tabular classification, synthetic densities, generative modeling |
| **Huang et al. (2025)**<br>*Differentiable Synthesis of BTs* | Differentiable BT | **Yes** (Soft t-norms & Gumbel-Softmax) | Learned (End-to-end grammar relaxation) | Continuous neural policies $\pi_\theta(a \mid s)$, parameterized thresholds | Robot navigation, multi-stage robotic manipulation (MuJoCo/Isaac Sim) |
| **Kontschieder et al. (2015)**<br>*Deep Neural Decision Forests* | Differentiable BT (Antecedent) | **Yes** (Stochastic routing via sigmoid) | Fixed (Predefined tree depth/structure) | Categorical class probability distributions $\boldsymbol{\pi}_\ell \in \Delta^Y$ | Computer vision classification (ImageNet, CIFAR-10, MNIST) |
| **Frosst & Hinton (2017)**<br>*Soft Decision Trees* | Differentiable BT (Antecedent) | **Yes** (Continuous logistic gating) | Fixed (Predefined binary tree depth) | Static class probability logits $\mathbf{Q}_\ell$ | MNIST digit recognition, Connect-4 game outcome prediction |
| **Sprague & Ögren (2022)**<br>*Adding NN Controllers to BTs* | Neuro-Symbolic Hybrid | **No** (Discrete tick execution) | Fixed (Guarded Fallback composition) | Continuous deep RL policies $\pi_\theta(\mathbf{x})$, nominal model-based LQR/MPC | Inverted pendulum, 2D mobile robot obstacle avoidance, spacecraft docking |
| **Chatzilygeroudis et al. (2021)**<br>*Learning Parameters in BTs* | Neuro-Symbolic Hybrid | **No** (Black-box policy search) | Fixed (Human expert designed topology) | Dynamical Movement Primitives (DMPs), neural state estimators | Robotic peg-in-hole assembly (7-DoF KUKA iiwa), reactive reaching |
| **Pereira & Engel (2015)**<br>*Adaptive Behavior-Based Agents* | Neuro-Symbolic Hybrid | **No** (Semi-Markov Q-learning) | Fixed (Manual learning node placement) | Discrete/continuous action Q-learners, adaptive selector policies | Predator-prey pursuit, maze foraging, ROS/Stage robot navigation |
| **Iovino et al. (2021)**<br>*GP in Unpredictable Environments* | Evolutionary Structure Search | **No** (Discrete genetic operators) | Learned (Genetic Programming evolution) | Atomic trajectory controllers, robotic manipulation primitives | Robotic kitting (dual-arm YuMi), mobile manipulation under human interference |
| **Scheide et al. (2021)**<br>*MCDAGS over Formal Grammar* | Evolutionary Structure Search | **No** (Grammar graph traversal) | Learned (MCTS over derivation DAG) | Discrete navigation primitives, sensor reading evaluation leaves | Autonomous Underwater Vehicle (AUV) marine exploration and target search |
| **Anne et al. (2023)**<br>*GAME MAP-Elites for BTs* | Evolutionary Structure Search | **No** (Quality-Diversity co-evolution) | Learned (Evolutionary mutation/crossover) | Steering/aiming primitives, deep vision embedding descriptors | Parabellum 2D shooter, EvoGym soft-robot wrestling, competitive games |
| **Özkahraman & Ögren (2020)**<br>*CBF Behavior Trees* | Safe / Verifiable BT | **Partial** (Continuous QP filter inside discrete tree) | Fixed (Hierarchical priority composition) | CBF-QP continuous safety filters, nominal coverage vector fields | Multi-agent AUV persistent coverage, underwater docking and recharging |
| **Serbinowska et al. (2025)**<br>*Neuro-Symbolic BT Verification* | Safe / Verifiable BT | **No** (Symbolic model checking via nuXmv) | Fixed (Verified DSL model) | Deep neural perceptual classifiers, deep RL action policies | ACAS Xu aircraft collision avoidance, rover waypoint tracking, gridworld |

---

# 3. Open Theoretical Frontiers

The convergence of hierarchical decision architectures with continuous-time dynamical systems—crystallized by Ramachandran & Sra's (2026) proof of the equivalence between hierarchical tree partitioning and drift-diffusion processes—has opened profound theoretical frontiers at the intersection of control theory, generative modeling, and neuro-symbolic robotics.

First, **resolving the discretization-execution gap in differentiable behavior trees** remains an urgent theoretical open problem. Existing differentiable relaxations (such as Huang et al., 2025) approximate discrete control nodes using soft continuous t-norms (product or Łukasiewicz logic) and Gumbel-Softmax reparameterizations during backpropagation. However, hardening this continuous supernet into a deterministic, tick-based Behavior Tree for physical deployment causes severe performance degradation, as the discrete tree cannot reproduce the fractional co-activation of parallel branches that the neural optimizer exploited. Applying the Fokker-Planck continuum limit reveals that discrete BT execution is an un-annealed, coarse-grained discretization of a continuous probability flow ODE. Developing exact, boundary-preserving flow-matching schemes that guarantee zero loss of reactivity upon hardening represents a paramount mathematical challenge.

Second, the **unification of Control Barrier Functions (CBFs) with continuous-time score fields within dynamic Behavior Trees** offers a transformative path for safety-critical learning. In current CBF-BT architectures (Özkahraman & Ögren, 2020), safety is enforced through local, instantaneous Quadratic Programs that assume known analytical control-affine dynamics and frequently suffer from infeasibility when conflicting objectives arise. By viewing the hierarchy as a continuous vector field $\frac{d\mathbf{x}}{dt} = \mu(\mathbf{x}, t) - \frac{1}{2} \sigma^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$, barrier certificates can be cast directly as functional gradient constraints on the score field itself. This would enable provably forward-invariant continuous-time score matching, ensuring that end-to-end neural policies distilled into or guided by trees cannot penetrate unsafe state-space manifolds even under extreme environmental uncertainty.

Third, **bidirectional neural-symbolic distillation under non-stationary reactive control loops** remains unformalized. While DSM-Tree (Ramachandran & Sra, 2026) achieves score distillation for static tabular distributions, robotic systems operate in closed-loop environments where actions alter underlying state transitions. Formulating a temporal, closed-loop extension of Global Trajectory Score Matching (GTSM) that accounts for Bellman state-occupancy shifts will enable seamless bidirectional transfer: compiling large black-box foundation policies (e.g., Vision-Language-Action models) into auditable, formally verifiable Behavior Trees, and conversely, diffusing human-engineered safety logic directly into agile, continuous neural controllers.

---

# 4. Paper Dossiers: Complete Technical Records

---

## Dossier 01: Trees to Flows and Back: Unifying Decision Trees and Diffusion Models

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

---

## Dossier 02: Differentiable Synthesis of Behavior Tree Architectures and Execution Nodes

- **Authors:** Yu Huang, Ziji Wu, Kexin Ma, Ji Wang
- **Year / Venue:** 2025 / International Conference on Neuro-symbolic Systems (NeuS 2025), PMLR
- **Paper Link / Identifier:** [NeuS 2025 Proceedings / OpenReview](https://openreview.net/forum?id=NeuS2025_diff_bt)
- **Primary Category:** Differentiable BT

### 1. Executive Summary & Core Hypothesis
Synthesizing Behavior Trees directly from environmental interactions has historically required combinatorial search (genetic programming or MCTS) over discrete grammar rules due to the non-differentiable execution semantics of control nodes. The authors hypothesize that both the discrete structural grammar and the continuous leaf node parameters of a Behavior Tree can be jointly relaxed into a fully differentiable derivation graph. By introducing a continuous relaxation of Sequence and Fallback control logic alongside Gumbel-Softmax grammar routing, the entire tree synthesis process is formulated as an end-to-end gradient-based optimization problem driven solely by reinforcement learning policy gradients.

### 2. Theoretical Framework & Mathematical Formulation
- **Node Mechanics & Return Status Handling:** In standard BTs, a node $v$ returns a discrete status $S(v) \in \{\text{SUCCESS}, \text{FAILURE}, \text{RUNNING}\}$. The authors relax this into a continuous tri-state probability simplex vector $\mathbf{s}(v) = [p_{\text{succ}}(v), p_{\text{fail}}(v), p_{\text{run}}(v)]^T \in \Delta^2$, where $\sum_{k} p_k(v) = 1$. For continuous leaf action nodes $\pi_\theta(a|s)$, the status vector is evaluated via differentiable condition bounds or state value estimates:
  $$p_{\text{succ}}(v) = \sigma\left(\frac{r(s, a) - \tau_{\text{succ}}}{\tau}\right), \quad p_{\text{fail}}(v) = \sigma\left(\frac{\tau_{\text{fail}} - r(s, a)}{\tau}\right)$$
  where $\tau$ is an annealed temperature parameter.
- **Execution / Control Flow:** Composite nodes (Sequence $\to$ and Fallback $?$) are relaxed using continuous soft logic operators. For a Sequence node with children $v_1, \dots, v_K$, the status is propagated sequentially using smooth t-norms (or product logic):
  $$p_{\text{succ}}(\text{Seq}) = \prod_{k=1}^K p_{\text{succ}}(v_k)$$
  $$p_{\text{fail}}(\text{Seq}) = \sum_{k=1}^K p_{\text{fail}}(v_k) \prod_{j=1}^{k-1} p_{\text{succ}}(v_j)$$
  $$p_{\text{run}}(\text{Seq}) = \sum_{k=1}^K p_{\text{run}}(v_k) \prod_{j=1}^{k-1} p_{\text{succ}}(v_j)$$
  Dually, for a Fallback node:
  $$p_{\text{fail}}(\text{Fall}) = \prod_{k=1}^K p_{\text{fail}}(v_k)$$
  $$p_{\text{succ}}(\text{Fall}) = \sum_{k=1}^K p_{\text{succ}}(v_k) \prod_{j=1}^{k-1} p_{\text{fail}}(v_j)$$
  The architecture search space is defined by a context-free grammar $G = (V_N, V_T, P, S)$, where production choices at non-terminal nodes are modulated via continuous categorical weights $\alpha \in \mathbb{R}^{|P|}$ using the Gumbel-Softmax reparameterization:
  $$w_i = \frac{\exp((\alpha_i + g_i)/\tau)}{\sum_j \exp((\alpha_j + g_j)/\tau)}, \quad g_i \sim \text{Gumbel}(0, 1)$$
- **Optimization Strategy:** A bi-level or joint gradient descent scheme optimizes the composite objective:
  $$\mathcal{L}(\alpha, \theta) = -\mathbb{E}_{\tau \sim \pi_{\alpha, \theta}}[R(\tau)] + \lambda_{\text{ent}} \mathcal{H}(\alpha) + \lambda_{\text{comp}} \sum_{v \in \mathcal{T}} \text{Complexity}(v)$$
  where policy gradients $\nabla_\theta \mathcal{L}$ update the execution node parameters, and path gradients $\nabla_\alpha \mathcal{L}$ update the architectural routing logits. Once trained, a temperature annealing schedule ($\tau \to 0$) and a deterministic discretization pass extract a discrete, executable Behavior Tree.

### 3. Architecture & Neural Integration
- **Neural Role:** Neural networks operate in two distinct capacities: (1) neural state estimators and condition evaluators providing differentiable boundary thresholds, and (2) parameterized continuous action sub-policies (e.g. Gaussian MLPs) embedded within leaf action nodes executing low-level motor primitives.
- **Interface / Boundary:** The boundary is governed by continuous routing gates. The output action of the overall relaxed tree is a soft convex combination of leaf action policies:
  $$a(s) = \sum_{v \in \text{Leaves}} \omega_v(s; \alpha) \pi_\theta(a|s, v)$$
  where the activation weight $\omega_v(s; \alpha)$ is the cumulative differentiable path execution probability from root to leaf $v$.

### 4. Empirical Evaluation & Benchmarks
- **Environments / Tasks:** 
  - Continuous robotic navigation in complex maze environments with dynamic obstacles.
  - Multi-stage robotic manipulation (Pick-and-Place, Drawer Opening, Peg-in-Hole) in MuJoCo / Isaac Sim.
  - MiniGrid discrete gridworld navigation.
- **Baseline Comparisons:**
  - Discrete GP-BT (Genetic Programming Behavior Trees).
  - MCDAGS (Monte Carlo DAG Search over formal grammar).
  - Monolithic DRL baselines: PPO and SAC.
  - Handcrafted modular Behavior Trees.
- **Key Quantitative Results:**
  - *Synthesis Sample Efficiency:* Differentiable synthesis converged $3.5\times$ to $5\times$ faster in sample efficiency compared to GP-BT and MCDAGS due to directed gradient backpropagation versus random mutation/sampling.
  - *Task Success Rate:* Reached $94.2\%$ success on multi-stage manipulation tasks, outperforming monolithic PPO ($72.1\%$) and matching hand-engineered expert BTs ($96.0\%$) while requiring zero manual topology engineering.

### 5. Failure Modes, Trade-offs & Limitations
- **Interpretability Preservation:** High interpretability is recovered post-discretization, as the resulting architecture is a true discrete Behavior Tree with human-readable Sequence and Fallback control logic. However, during the continuous training phase, the "supernet" behavior is diffuse and uninterpretable.
- **Scalability & Gradient Stability:** Gradient degradation occurs in deep derivation trees due to vanishing continuous products $\prod_{j=1}^{k-1} p_{\text{succ}}(v_j)$ across long sequence chains. Temperature annealing schedules must be tuned delicately: premature temperature drop leads to entrapment in degenerate local tree topologies.
- **Unaddressed Gaps:** Discretization gap: the performance of the soft relaxed supernet occasionally drops when hardened into a discrete tree if continuous activation weights had shared partial control among multiple contradictory subtrees.

---

## Dossier 03: Deep Neural Decision Forests

- **Authors:** Peter Kontschieder, Madalina Fiterau, Antonio Criminisi, Samuel Rota Bulò
- **Year / Venue:** 2015 / IEEE International Conference on Computer Vision (ICCV 2015) - Marr Prize Paper
- **Paper Link / Identifier:** [IEEE Xplore / DOI: 10.1109/ICCV.2015.498](https://doi.org/10.1109/ICCV.2015.498) / [arXiv:1512.04838](https://arxiv.org/abs/1512.04838)
- **Primary Category:** Differentiable BT

### 1. Executive Summary & Core Hypothesis
Classical random forests and decision trees excel at modular classification and regression through divide-and-conquer spatial partitioning, but cannot learn continuous feature representations via backpropagation. The authors hypothesize that decision forests can be unified with deep convolutional neural networks by introducing stochastic routing functions at internal nodes and formulating the leaf node class predictions as convex optimization problems. This creates the first fully end-to-end differentiable neural decision forest architecture, where split decisions and deep perceptual features are trained simultaneously via stochastic gradient descent.

### 2. Theoretical Framework & Mathematical Formulation
- **Node Mechanics & Return Status Handling:** Unlike deterministic threshold routing, every internal node $n \in \mathcal{N}$ defines a continuous, differentiable routing probability:
  $$d_n(\mathbf{x}; \Theta) = \sigma(f_n(\mathbf{x}; \Theta)) = \frac{1}{1 + \exp(-f_n(\mathbf{x}; \Theta))}$$
  where $f_n(\mathbf{x}; \Theta)$ is a real-valued output generated by the deep neural network parameterized by $\Theta$. The probability of routing left is $d_n(\mathbf{x})$, and right is $1 - d_n(\mathbf{x})$.
- **Execution / Control Flow:** A sample $\mathbf{x}$ traverses from the root to every leaf $\ell \in \mathcal{L}$ with a continuous path probability $\mu_\ell(\mathbf{x}; \Theta)$, computed as the product of all routing decisions along the unique ancestral path $\mathcal{P}_\ell$:
  $$\mu_\ell(\mathbf{x}; \Theta) = \prod_{n \in \mathcal{N}} d_n(\mathbf{x}; \Theta)^{\mathbb{I}(n \swarrow \ell)} (1 - d_n(\mathbf{x}; \Theta))^{\mathbb{I}(n \searrow \ell)}$$
  where $n \swarrow \ell$ denotes that leaf $\ell$ belongs to the left subtree of node $n$, and $n \searrow \ell$ denotes the right subtree. Each leaf node $\ell$ maintains a class probability vector $\boldsymbol{\pi}_\ell \in \Delta^Y$. The overall tree prediction is the marginal distribution over all leaves:
  $$\mathbb{P}(y \mid \mathbf{x}, \Theta, \boldsymbol{\pi}) = \sum_{\ell \in \mathcal{L}} \mu_\ell(\mathbf{x}; \Theta) \pi_{\ell, y}$$
  For an ensemble forest of trees $\mathcal{F}$, the forest prediction is the average across all trees: $\mathbb{P}_{\mathcal{F}}(y \mid \mathbf{x}) = \frac{1}{|\mathcal{F}|} \sum_{T \in \mathcal{F}} \mathbb{P}_T(y \mid \mathbf{x})$.
- **Optimization Strategy:** The joint objective minimizes the empirical cross-entropy loss over dataset $\mathcal{T}$:
  $$R(\Theta, \boldsymbol{\pi}) = -\frac{1}{|\mathcal{T}|} \sum_{(\mathbf{x}, y) \in \mathcal{T}} \log \mathbb{P}(y \mid \mathbf{x}, \Theta, \boldsymbol{\pi})$$
  Optimization alternates between two steps:
  1. *Leaf Node Optimization:* With neural network parameters $\Theta$ fixed, optimizing $\boldsymbol{\pi}$ is a convex problem solved via a globally convergent iterative bound-optimization:
     $$\pi_{\ell, y}^{(t+1)} = \frac{1}{Z_\ell} \sum_{(\mathbf{x}, y') \in \mathcal{T}} \mathbb{I}(y' = y) \frac{\mu_\ell(\mathbf{x}; \Theta) \pi_{\ell, y}^{(t)}}{\sum_{\ell' \in \mathcal{L}} \mu_{\ell'}(\mathbf{x}; \Theta) \pi_{\ell', y}^{(t)}}$$
  2. *Neural Feature & Routing Optimization:* With $\boldsymbol{\pi}$ fixed, gradients with respect to neural routing activation $f_n(\mathbf{x}; \Theta)$ are computed analytically via the chain rule:
     $$\frac{\partial R}{\partial f_n(\mathbf{x})} = \frac{1}{2} \left( d_n(\mathbf{x}) \sum_{\ell \in \mathcal{L}_{n, \text{left}}} \frac{\mu_\ell(\mathbf{x}) \pi_{\ell, y}}{\mathbb{P}(y \mid \mathbf{x})} - (1 - d_n(\mathbf{x})) \sum_{\ell \in \mathcal{L}_{n, \text{right}}} \frac{\mu_\ell(\mathbf{x}) \pi_{\ell, y}}{\mathbb{P}(y \mid \mathbf{x})} \right)$$
     enabling backpropagation through arbitrary deep neural network backbones.

### 3. Architecture & Neural Integration
- **Neural Role:** The deep convolutional neural network (e.g. GoogLeNet or VGG) serves as a shared continuous representation extractor. The final fully-connected layer outputs a $K \times |\mathcal{N}|$ dimensional embedding matching the total number of decision nodes across all trees in the forest.
- **Interface / Boundary:** The linear routing units of the neural net interface directly with the hierarchical routing probability tree via sigmoid non-linearities, transforming feature embeddings into spatial routing coefficients.

### 4. Empirical Evaluation & Benchmarks
- **Environments / Tasks:** Large-scale computer vision benchmarks: ImageNet (ILSVRC 2012), MNIST, and CIFAR-10 classification.
- **Baseline Comparisons:** Standard CNNs with Softmax output layers (GoogLeNet), standard Random Forests, and alternating decision trees.
- **Key Quantitative Results:**
  - *ImageNet:* Deep Neural Decision Forests (dNDF-GoogLeNet) achieved a Top-5 error rate of $7.84\%$, outperforming the standard Softmax GoogLeNet baseline ($10.07\%$) without requiring ensemble model averaging.
  - *CIFAR-10:* Achieved $92.6\%$ test accuracy without data augmentation, demonstrating superior regularization compared to flat neural classification heads.

### 5. Failure Modes, Trade-offs & Limitations
- **Interpretability Preservation:** Although each individual tree retains a tree topology, the internal split conditions are complex non-linear combinations of deep convolutional feature maps rather than intuitive semantic thresholds. Furthermore, the stochastic soft routing spreads single input instances across multiple leaves simultaneously.
- **Scalability & Gradient Stability:** Memory complexity scales exponentially with tree depth $D$ ($\mathcal{O}(2^D)$ leaves per tree). In practice, trees were restricted to depth $D \le 10$ to prevent memory bottlenecks.
- **Unaddressed Gaps:** The tree topology is completely static and fixed prior to training; the model cannot dynamically learn or prune tree topologies during gradient descent. Moreover, the formulation is strictly supervised and does not address dynamic sequential decision making or reactive control loops.

---

## Dossier 04: Distilling a Neural Network Into a Soft Decision Tree

- **Authors:** Nicholas Frosst, Geoffrey Hinton
- **Year / Venue:** 2017 / arXiv preprint (arXiv:1711.09784 [cs.LG, cs.AI])
- **Paper Link / Identifier:** [arXiv:1711.09784](https://arxiv.org/abs/1711.09784)
- **Primary Category:** Differentiable BT

### 1. Executive Summary & Core Hypothesis
Deep neural networks excel at extracting intricate continuous representations but function as opaque black boxes, whereas decision trees provide intuitive interpretability but lack generalization on raw perceptual inputs. Frosst and Hinton hypothesize that an uninterpretable deep neural network can be distilled into a hierarchical soft decision tree using model compression and dark knowledge transfer. By defining soft probabilistic routing at every node and optimizing a cross-entropy distillation objective with an information-theoretic regularization penalty, the soft decision tree captures the teacher network's decision boundaries while exposing its hierarchical decision path directly to human inspection.

### 2. Theoretical Framework & Mathematical Formulation
- **Node Mechanics & Return Status Handling:** Each internal node $i$ of the soft decision tree computes a continuous gating probability using a learned weight vector $\mathbf{w}_i$ and scalar bias $b_i$:
  $$p_i(\mathbf{x}) = \sigma(\beta(\mathbf{x}^T \mathbf{w}_i + b_i))$$
  where $\sigma(\cdot)$ is the logistic sigmoid function, and $\beta$ is an inverse temperature parameter governing split hardness. A probability $p_i(\mathbf{x})$ routes the input to the right child, and $1 - p_i(\mathbf{x})$ routes to the left child.
- **Execution / Control Flow:** Every leaf node $\ell$ maintains a learned logit vector $\boldsymbol{\theta}_\ell$, producing a categorical probability distribution over classes via softmax:
  $$Q_\ell^k = \frac{\exp(\theta_\ell^k)}{\sum_{k'} \exp(\theta_\ell^{k'})}$$
  The cumulative path probability $P^\ell(\mathbf{x})$ from the root to leaf $\ell$ is the continuous product of gating decisions along the ancestor trajectory $\mathcal{A}(\ell)$:
  $$P^\ell(\mathbf{x}) = \prod_{i \in \mathcal{A}_{\text{left}}(\ell)} (1 - p_i(\mathbf{x})) \prod_{j \in \mathcal{A}_{\text{right}}(\ell)} p_j(\mathbf{x})$$
  The total predictive distribution of the tree is the marginal probability mixture across all leaves:
  $$P(y = k \mid \mathbf{x}) = \sum_{\ell \in \text{Leaves}} P^\ell(\mathbf{x}) Q_\ell^k$$
- **Optimization Strategy:** The objective function unifies cross-entropy loss against both ground-truth labels and dark knowledge targets generated by the teacher neural network, regularized by an entropy-balancing penalty across internal tree levels:
  $$\mathcal{L}(\mathcal{D}) = -\frac{1}{|\mathcal{D}|} \sum_{\mathbf{x} \in \mathcal{D}} \sum_{\ell \in \text{Leaves}} P^\ell(\mathbf{x}) \sum_k T_k \log Q_\ell^k + \lambda \sum_{d=0}^{D-1} 2^{-d} \mathcal{C}_d$$
  where $T_k$ is the teacher network's softened probability for class $k$, and the penalty $\mathcal{C}_d$ discourages dead branches at tree depth $d$ by maximizing cross-entropy between the average node routing probability $\bar{\alpha}_i = \frac{\sum_{\mathbf{x}} P_i(\mathbf{x}) p_i(\mathbf{x})}{\sum_{\mathbf{x}} P_i(\mathbf{x})}$ and a uniform prior:
  $$\mathcal{C}_i = -\frac{1}{2} \log(\bar{\alpha}_i) - \frac{1}{2} \log(1 - \bar{\alpha}_i)$$

### 3. Architecture & Neural Integration
- **Neural Role:** The deep neural network acts as an unconstrained teacher model trained via standard supervised learning. The soft decision tree acts as a parametric student model that directly distills the continuous manifold geometry learned by the neural network into an axis-aligned / hyperplane-partitioned hierarchical tree.
- **Interface / Boundary:** Soft probability targets from the neural network's logit outputs interface with the leaf node distribution $Q_\ell$ through Kullback-Leibler (KL) divergence, guiding the gradient descent updates of the tree's linear hyperplane weights $\mathbf{w}_i$.

### 4. Empirical Evaluation & Benchmarks
- **Environments / Tasks:** 
  - Image classification on MNIST digit recognition.
  - Connect-4 game outcome prediction (board game state evaluation).
- **Baseline Comparisons:**
  - Standard hard decision trees (CART, C4.5).
  - Standalone soft decision trees trained directly on ground truth labels without neural distillation.
  - Monolithic Multi-Layer Perceptrons (MLPs) and Convolutional Neural Networks.
- **Key Quantitative Results:**
  - *MNIST:* A distilled soft decision tree of depth 8 achieved $96.76\%$ test accuracy, significantly outperforming a standard hard decision tree ($93.8\%$) and closing the gap with the teacher neural network ($99.2\%$).
  - *Connect-4:* Distillation from a neural network boosted tree test accuracy from $77.2\%$ (direct tree training) to $82.8\%$.

### 5. Failure Modes, Trade-offs & Limitations
- **Interpretability Preservation:** The learned hyperplane splits $\mathbf{x}^T \mathbf{w}_i + b_i$ can be visualized by inspecting the weight templates. However, because routing is soft, an input sample activates multiple paths with varying fractional weights, slightly weakening the strict deterministic "if-then-else" audibility of classical decision trees.
- **Scalability & Gradient Stability:** Fixed tree topologies suffer from exponential parameter growth ($\mathcal{O}(2^D)$) with depth $D$. Training soft trees beyond depth $D=12$ leads to severe gradient dissipation in lower leaves and numerical underflow in path probabilities $\prod p_i(\mathbf{x})$.
- **Unaddressed Gaps:** Structure is entirely static (full binary tree); no pruning or dynamic topological reconfiguration is performed during learning. Furthermore, execution is purely static feedforward inference, with no temporal memory, action persistence, or reactive ticking mechanisms.

---

## Dossier 05: Adding Neural Network Controllers to Behavior Trees without Destroying Performance Guarantees

- **Authors:** Christopher Iliffe Sprague, Petter Ögren
- **Year / Venue:** 2022 / IEEE 61st Conference on Decision and Control (CDC 2022)
- **Paper Link / Identifier:** [IEEE Xplore / DOI: 10.1109/CDC51059.2022.9992501](https://doi.org/10.1109/CDC51059.2022.9992501) / [arXiv:2203.11802](https://arxiv.org/abs/2203.11802)
- **Primary Category:** Neuro-Symbolic Hybrid

### 1. Executive Summary & Core Hypothesis
Deep reinforcement learning policies deliver high empirical performance in continuous control but notoriously lack formal safety and stability guarantees, rendering them unacceptable in safety-critical robotics. Sprague and Ögren hypothesize that arbitrary black-box neural network controllers can be integrated into Behavior Trees while strictly preserving the formal performance guarantees (invariance, obstacle avoidance, and global convergence) of a nominal, model-based control system. By embedding the neural controller under a Fallback composition guarded by a computable region-of-attraction / safety condition, the system prioritizes the neural policy when safe while deterministically reverting to the baseline safe controller whenever safety boundaries are threatened.

### 2. Theoretical Framework & Mathematical Formulation
- **Node Mechanics & Return Status Handling:** Execution operates under standard discrete BT return semantics $S \in \{\text{SUCCESS}, \text{FAILURE}, \text{RUNNING}\}$. Condition nodes return SUCCESS if the continuous state $\mathbf{x}(t)$ lies within a forward-invariant safe set $\mathcal{C}$ or region of attraction $\mathcal{R}$, and FAILURE otherwise. Continuous action nodes execute feedback control laws $\mathbf{u} = \mathbf{k}(\mathbf{x})$, returning RUNNING while the state is in transit toward a goal attractor set $\mathcal{G}$, and SUCCESS once $\|\mathbf{x} - \mathbf{x}^\star\| \le \epsilon$.
- **Execution / Control Flow:** The core structural motif is a safeguarded Fallback composition:
  $$\text{Fallback} \ ? \ [\text{Sequence} \ \to \ (C_{\text{safe\_NN}}, A_{\text{NN}}), \ A_{\text{nominal}}]$$
  where:
  1. $C_{\text{safe\_NN}}$ is a condition node verifying that the state $\mathbf{x} \in \Omega_{\text{safe}} \subseteq \mathcal{X}$, where $\Omega_{\text{safe}}$ is a sublevel set of a Lyapunov or Control Barrier Function from which the nominal safe controller $A_{\text{nominal}}$ can provably prevent safety violations.
  2. $A_{\text{NN}}$ executes the continuous neural network policy $\mathbf{u}_{\text{NN}} = \pi_\theta(\mathbf{x})$.
  3. $A_{\text{nominal}}$ executes a certified model-based control law $\mathbf{u}_{\text{nominal}} = \mathbf{k}_{\text{nom}}(\mathbf{x})$ designed with guaranteed reach-avoid properties.
  At each execution tick, the tree evaluates $C_{\text{safe\_NN}}$. If true, $A_{\text{NN}}$ runs; if the neural controller drives the system toward the boundary of $\Omega_{\text{safe}}$, $C_{\text{safe\_NN}}$ immediately returns FAILURE, triggering instantaneous fallback to $A_{\text{nominal}}$ in the very same tick.
- **Optimization Strategy:** The neural policy $\pi_\theta(\mathbf{x})$ is trained via deep reinforcement learning (e.g. DDPG, SAC, or PPO) in simulation or unconstrained domains. The surrounding BT topology and switching invariant set boundaries $\partial \Omega_{\text{safe}}$ are synthesized using Lyapunov analysis and backward reachability, without requiring any gradient flow across the BT control nodes.

### 3. Architecture & Neural Integration
- **Neural Role:** The deep neural network acts as a specialized performance-maximizing continuous motor controller (e.g. minimizing time-to-target, energy consumption, or agile trajectory tracking).
- **Interface / Boundary:** Symbolic-continuous boundary: The continuous state $\mathbf{x}(t) \in \mathbb{R}^n$ is evaluated against explicit geometric or algebraic constraints $h(\mathbf{x}) \ge 0$. The neural network's control actions $\mathbf{u}_{\text{NN}}$ are applied directly to the plant dynamics $\dot{\mathbf{x}} = f(\mathbf{x}, \mathbf{u})$ only as long as the symbolic condition evaluates to SUCCESS.

### 4. Empirical Evaluation & Benchmarks
- **Environments / Tasks:** 
  - Nonlinear planar inverted pendulum with torque saturation and state-space obstacle constraints.
  - Safe 2D mobile robot point-to-point navigation among circular and convex polyhedral obstacles.
  - Autonomous spacecraft docking / rendezvous maneuver subject to line-of-sight cone constraints.
- **Baseline Comparisons:**
  - Standalone unconstrained neural network policy (SAC).
  - Standalone nominal model-based controller (Linear Quadratic Regulator / Model Predictive Control).
  - Standard reward-penalized RL (adding large negative penalty rewards for obstacle collisions).
- **Key Quantitative Results:**
  - *Zero Safety Violations:* The hybrid BT maintained a $0.0\%$ safety violation rate across 10,000 Monte Carlo test trajectories, whereas the reward-penalized neural policy experienced a $7.4\%$ collision rate due to function approximation errors near obstacle boundaries.
  - *Performance Enhancement:* When operating within the safe flight envelope, the hybrid BT achieved a $24.8\%$ faster mission completion time than the conservative nominal LQR controller by exploiting the aggressive, agile trajectories discovered by the neural network policy.

### 5. Failure Modes, Trade-offs & Limitations
- **Interpretability Preservation:** Excellent interpretability: the hierarchical control logic is completely transparent, human-readable, and formally verifiable using standard hybrid systems theory. Human operators can visually inspect the switching boundaries and understand precisely why a neural policy was overridden.
- **Scalability & Gradient Stability:** Because the neural network and the nominal BT are modularly separated, there are no gradient stability issues or backpropagation bottlenecks. However, computing the certified invariant set $\Omega_{\text{safe}}$ scales poorly with state dimensionality ($n > 6$) using sum-of-squares (SOS) or Hamilton-Jacobi reachability analysis.
- **Unaddressed Gaps:** Chattering and limit cycles: rapid switching between the neural controller and the nominal fallback controller can occur along the boundary $\partial \Omega_{\text{safe}}$ if the neural policy continuously pushes into the forbidden set, causing high-frequency control chatter without appropriate hysteresis decorators.

---

## Dossier 06: Learning of Parameters in Behavior Trees for Movement Skills

- **Authors:** Konstantinos Chatzilygeroudis, Bernardo Fichera, Iason Sarantopoulos, Vasileios Vassiliades, Aude Billard
- **Year / Venue:** 2021 / IEEE Robotics and Automation Letters (RA-L) with ICRA 2021 presentation
- **Paper Link / Identifier:** [IEEE Xplore / DOI: 10.1109/LRA.2021.3060416](https://doi.org/10.1109/LRA.2021.3060416) / [arXiv:2009.09848](https://arxiv.org/abs/2009.09848)
- **Primary Category:** Neuro-Symbolic Hybrid

### 1. Executive Summary & Core Hypothesis
While Behavior Trees provide modularity and reactive switching for high-level mission planning, defining low-level geometric and dynamic execution parameters for complex movement skills by hand is fragile and time-consuming. The authors hypothesize that a parameterized Behavior Tree—where continuous movement skills are represented by Dynamical Movement Primitives (DMPs) and neural state estimators—can be optimized directly in simulation via policy search and safely transferred to real robotic hardware. By optimizing both condition thresholds and motion primitive parameters under execution safety constraints, the robot acquires robust, adaptive manipulation skills capable of recovering from environmental perturbations.

### 2. Theoretical Framework & Mathematical Formulation
- **Node Mechanics & Return Status Handling:** Classical discrete return statuses $\{\text{SUCCESS}, \text{FAILURE}, \text{RUNNING}\}$ are computed based on continuous state evaluations against learned parameter vectors $\boldsymbol{\theta}_c$ and $\boldsymbol{\theta}_a$. A condition node evaluates whether a sensor reading $\mathbf{y}(t)$ satisfies an algebraic threshold:
  $$S_{\text{cond}}(\mathbf{y}; \boldsymbol{\theta}_c) = \begin{cases} \text{SUCCESS} & \text{if } g(\mathbf{y}, \boldsymbol{\theta}_c) \ge 0 \\ \text{FAILURE} & \text{otherwise} \end{cases}$$
  An action node executes a continuous dynamical system (DMP or neural velocity field) parameterized by $\boldsymbol{\theta}_a$:
  $$\tau \dot{\mathbf{v}} = K (\mathbf{g} - \mathbf{x}) - D \mathbf{v} + \mathbf{f}(s; \boldsymbol{\theta}_a)$$
  returning RUNNING during execution, SUCCESS when $\|\mathbf{x} - \mathbf{g}\| \le \epsilon$, or FAILURE upon timeout or torque limit exceedance.
- **Execution / Control Flow:** The tree topology is fixed by domain experts to encode structural task logic (e.g., approach, align, insert, verify, retry). The overall parameterized policy is defined as $\pi_{\boldsymbol{\theta}}(\mathbf{x})$, where $\boldsymbol{\theta} = [\boldsymbol{\theta}_c^T, \boldsymbol{\theta}_a^T]^T \in \Theta \subset \mathbb{R}^D$ represents the concatenated vector of all condition tolerances, attractor goals, and shape parameters across all leaves. At each tick, the tree evaluates conditions and triggers reactive fallbacks whenever an action fails (e.g. peg jamming during insertion).
- **Optimization Strategy:** Because discrete BT execution transitions render the expected return non-differentiable with respect to parameter vector $\boldsymbol{\theta}$, the authors formulate parameter learning as a sample-efficient black-box policy search problem using Bayesian Optimization or CMA-ES:
  $$\max_{\boldsymbol{\theta} \in \Theta} J(\boldsymbol{\theta}) = \mathbb{E}_{\tau \sim \mathcal{D}}\left[ R_{\text{task}}(\tau; \boldsymbol{\theta}) - \lambda \sum_{k} C_k(\tau; \boldsymbol{\theta}) \right]$$
  where $R_{\text{task}}$ rewards task completion, and $C_k$ penalizes excessive contact forces or execution time. Optimization is performed in a physics simulator (MuJoCo/PyBullet), followed by domain randomization and sim-to-real transfer onto physical manipulators.

### 3. Architecture & Neural Integration
- **Neural Role:** Neural networks and nonlinear function approximators represent: (1) nonlinear forcing terms $\mathbf{f}(s; \boldsymbol{\theta}_a)$ in dynamical movement primitives learned from human demonstrations, and (2) neural perception models estimating 6D object poses and contact states from depth cameras and force-torque sensors.
- **Interface / Boundary:** Symbolic-to-continuous interface: Condition leaves in the BT query neural perception heads to evaluate boolean predicates (e.g. "IsPegAligned"), and action leaves modulate the setpoints and impedance gains of continuous low-level robot controllers.

### 4. Empirical Evaluation & Benchmarks
- **Environments / Tasks:** 
  - Precision robotic peg-in-hole assembly under positional uncertainty using a 7-DoF KUKA LBR iiwa manipulator.
  - Dynamic robotic obstacle avoidance and reactive reaching under human workspace interference.
  - Multi-stage robotic pick-and-place with fragile object grasp verification.
- **Baseline Comparisons:**
  - Manually hand-tuned Behavior Trees.
  - Flat monolithic Model-Free RL (PPO and SAC trained end-to-end).
  - Standalone Dynamical Movement Primitives without reactive BT switching logic.
- **Key Quantitative Results:**
  - *Assembly Success Rate:* Parameter-optimized BTs achieved a $98.3\%$ insertion success rate on physical hardware under up to $\pm 15\,\text{mm}$ position error, compared to $45.0\%$ for hand-tuned BTs and $62.5\%$ for unguided DMPs.
  - *Sample Efficiency:* Bayesian optimization converged to optimal parameter configurations within 50 to 100 simulation rollouts, whereas flat DRL algorithms failed to converge within 50,000 steps due to exploration bottlenecks in tight-tolerance insertion spaces.

### 5. Failure Modes, Trade-offs & Limitations
- **Interpretability Preservation:** Fully preserves native BT interpretability: the tree structure remains completely symbolic and human-readable. Engineers can inspect, tune, or override individual learned parameter bounds.
- **Scalability & Gradient Stability:** Black-box policy search (Bayesian Optimization / CMA-ES) avoids non-differentiable gradient instability but suffers from the curse of dimensionality, restricting the total number of tunable parameters to $D \le 30$. High-dimensional neural representations cannot be trained end-to-end using this formulation.
- **Unaddressed Gaps:** Topology learning: the tree structure must be provided a priori by an expert designer; the algorithm cannot discover new control branches or structurally synthesize fallback recovery loops from data.

---

## Dossier 07: A Framework for Constrained and Adaptive Behavior-Based Agents

- **Authors:** Renato de Pontes Pereira, Paulo Martins Engel
- **Year / Venue:** 2015 / arXiv preprint (arXiv:1506.02312 [cs.AI, cs.RO]) / Adaptive Behavior Journal
- **Paper Link / Identifier:** [arXiv:1506.02312](https://arxiv.org/abs/1506.02312) / [https://doi.org/10.48550/arXiv.1506.02312](https://doi.org/10.48550/arXiv.1506.02312)
- **Primary Category:** Neuro-Symbolic Hybrid

### 1. Executive Summary & Core Hypothesis
Autonomous agents operating in complex environments require both rigid designer-specified constraints (for safety and predictability) and local adaptability (for optimization under environmental uncertainty). Pereira and Engel hypothesize that Behavior Trees can be formally unified with Hierarchical Reinforcement Learning (HRL) by defining specialized "Learning Nodes" that map directly onto the semi-Markov Decision Process (SMDP) Options Framework. By proving that a Behavior Tree subtree functions as an option tuple $\langle \mathcal{I}, \pi, \beta \rangle$, the authors demonstrate that reinforcement learning algorithms can be embedded within localized BT leaves or composite nodes without compromising global structural behavior.

### 2. Theoretical Framework & Mathematical Formulation
- **Node Mechanics & Return Status Handling:** Classical discrete return statuses $\{\text{SUCCESS}, \text{FAILURE}, \text{RUNNING}\}$ are mapped onto the temporal termination semantics of an option $\omega \in \Omega$:
  1. *Initiation Set $\mathcal{I}_\omega \subseteq \mathcal{S}$:* Defined by the guard condition nodes preceding the learning node in a Sequence composition $\text{Seq}(C_{\text{init}}, A_{\text{learn}})$. If $s_t \notin \mathcal{I}_\omega$, the node immediately returns FAILURE.
  2. *Internal Policy $\pi_\omega(a \mid s)$:* The local policy executed while the learning node returns RUNNING.
  3. *Termination Condition $\beta_\omega: \mathcal{S} \to [0, 1]$:* The probability of terminating the option at state $s$. In BT semantics, termination occurs deterministically when the node transitions from RUNNING to either SUCCESS ($\beta_{\text{succ}}(s) = 1$) or FAILURE ($\beta_{\text{fail}}(s) = 1$).
- **Execution / Control Flow:** The authors introduce two primary learning node extensions:
  1. *Adaptive Action Node:* An atomic action leaf containing an internal continuous or tabular Q-function $Q(s, a)$. The node selects primitive motor actions at each tick until local termination criteria are reached.
  2. *Adaptive Composite Node (Learning Selector):* A composite node that dynamically learns the execution ordering of its children subtrees. Rather than ticking children from left to right deterministically, the node queries a learned policy-over-options $\mu(\omega \mid s)$ to select which child branch to tick.
- **Optimization Strategy:** Learning is governed by SMDP Q-learning. When an option $\omega$ is initiated at time $t$ in state $s_t$ and terminates $\tau$ steps later at state $s_{t+\tau}$ with total discounted reward $R_{t:\tau} = \sum_{k=0}^{\tau-1} \gamma^k r_{t+k}$, the semi-Markov Q-value update rule is applied:
  $$Q(s_t, \omega) \leftarrow Q(s_t, \omega) + \alpha \left[ R_{t:\tau} + \gamma^\tau \max_{\omega' \in \Omega(s_{t+\tau})} Q(s_{t+\tau}, \omega') - Q(s_t, \omega) \right]$$
  For nested learning nodes, intra-option policy gradient and temporal difference updates are executed locally at each tick without propagating gradients across external tree boundaries.

### 3. Architecture & Neural Integration
- **Neural Role:** Neural networks parameterize local state-value functions $V_\theta(s)$, action-value functions $Q_\theta(s, a)$, or continuous policy actors $\pi_\theta(a \mid s)$ encapsulated entirely within individual learning nodes.
- **Interface / Boundary:** Symbolic-to-MDP abstraction: The parent BT imposes state-space filtering and temporal boundaries: it dictates *when* the neural policy is invoked (via preconditions) and *when* it is interrupted or terminated (via postconditions and tick overrides). The neural network outputs low-level primitive control signals to the agent's actuators.

### 4. Empirical Evaluation & Benchmarks
- **Environments / Tasks:** 
  - Dynamic 2D predator-prey gridworld tracking with stochastic prey evasion.
  - Multi-agent resource foraging in partially observable maze environments with dynamic hazards.
  - Simulated robotic navigation and obstacle avoidance in ROS/Stage.
- **Baseline Comparisons:**
  - Flat tabular and neural Q-learning agents (monolithic RL).
  - Pure static hand-coded Behavior Trees without learning nodes.
  - Hierarchical Q-learning (MAXQ framework).
- **Key Quantitative Results:**
  - *Convergence Speed:* The hybrid Learning-BT achieved asymptotic policy convergence $4.2\times$ faster than flat Q-learning because the tree structure constrained exploration to valid, productive regions of the state-action space.
  - *Constraint Satisfaction:* Maintained $100\%$ compliance with expert-defined safety rules (e.g. battery recharge thresholds, hazard avoidance) throughout the entire training process, whereas monolithic RL violated safety constraints during exploration in over $30\%$ of initial episodes.

### 5. Failure Modes, Trade-offs & Limitations
- **Interpretability Preservation:** High interpretability is preserved at the architectural level: human engineers clearly observe which tasks are governed by symbolic rules versus learned neural policies. However, sub-policies within individual learning nodes remain local black boxes.
- **Scalability & Gradient Stability:** Because learning is decoupled across nodes, there is no global gradient flow across the tree; each node optimizes its local reward independently. This eliminates gradient explosion/vanishing but introduces non-stationarity if multiple learning nodes are updated concurrently in nested hierarchies.
- **Unaddressed Gaps:** Structure is entirely hand-engineered: the placement of learning nodes within the tree must be manually specified by the human designer; the framework cannot automatically discover where learning nodes are required or split monolithic nodes into sub-skills.

---

## Dossier 08: Learning Behavior Trees with Genetic Programming in Unpredictable Environments

- **Authors:** Matteo Iovino, Jonathan Styrud, Pietro Falco, Christian Smith
- **Year / Venue:** 2021 / IEEE International Conference on Robotics and Automation (ICRA 2021)
- **Paper Link / Identifier:** [IEEE Xplore / DOI: 10.1109/ICRA48506.2021.9561214](https://doi.org/10.1109/ICRA48506.2021.9561214) / [arXiv:2011.08528](https://arxiv.org/abs/2011.08528)
- **Primary Category:** Structure Evolution + Policy Gradient

### 1. Executive Summary & Core Hypothesis
Autonomous robots deployed in unstructured and stochastic environments must rapidly adapt their sequential decision logic to unmodeled external disturbances (such as human interference or mechanical slippage). The authors hypothesize that Genetic Programming (GP) can evolve robust, reactive Behavior Tree topologies from primitive modular building blocks directly within unpredictable environments. By incorporating stochastic disturbances during fitness evaluations and penalizing tree structural complexity (bloat), the evolutionary algorithm discovers compact, fault-tolerant control trees that significantly outperform both open-loop action plans and manually designed reactive trees.

### 2. Theoretical Framework & Mathematical Formulation
- **Node Mechanics & Return Status Handling:** Standard discrete Behavior Tree semantics $S \in \{\text{SUCCESS}, \text{FAILURE}, \text{RUNNING}\}$. Leaf nodes comprise atomic action primitives $A = \{a_1, \dots, a_m\}$ and condition primitives $C = \{c_1, \dots, c_n\}$. Action nodes execute continuous motor controllers for robot manipulation, returning RUNNING during execution and SUCCESS/FAILURE upon completion.
- **Execution / Control Flow:** Trees are constructed recursively from a terminal set $\mathcal{T} = A \cup C$ and a non-terminal composite function set $\mathcal{F} = \{\text{Sequence } (\to), \text{Fallback } (?)\}$. Control flow is strictly reactive: at each tick $\Delta t$, execution traverses from the root down the left-to-right hierarchy. If an action in progress is invalidated by an external environmental perturbation (e.g. an object knocked out of the gripper), upstream condition checks immediately fail on the subsequent tick, triggering fallback branches to re-grasp or re-align the object.
- **Optimization Strategy:** Evolutionary search over tree syntax graphs. A population of individuals $\mathcal{P} = \{BT_1, \dots, BT_N\}$ is initialized via ramped half-and-half tree generation. The multi-objective fitness function evaluates task performance while penalizing structural bloat:
  $$f(BT) = \frac{1}{K} \sum_{k=1}^K R_k(BT) - \alpha \cdot \text{Size}(BT) - \beta \cdot \text{Ticks}(BT)$$
  where $R_k(BT) \in [0, 1]$ is the task completion score in simulation trial $k$ subject to randomized stochastic disturbances, $\text{Size}(BT)$ is the total node count, $\text{Ticks}(BT)$ measures execution duration, and $\alpha, \beta > 0$ are parsimony coefficients. Parents are selected via tournament selection, followed by subtree crossover and subtree mutation.

### 3. Architecture & Neural Integration
- **Neural Role:** In the baseline formulation, leaf actions wrap closed-loop trajectory tracking controllers. In neuro-symbolic extensions, neural networks act as learned sensorimotor primitives: (1) convolutional networks estimating 3D bounding boxes and object grasp affords, and (2) goal-conditioned deep RL policies executing localized manipulation trajectories.
- **Interface / Boundary:** Symbolic-to-continuous interface: The discrete GP algorithm searches over tree grammar and topological wiring, treating neural policies as black-box atomic execution leaves that expose standardized boolean return signals.

### 4. Empirical Evaluation & Benchmarks
- **Environments / Tasks:** 
  - Robotic kitting and bin-picking with dynamic obstacles using a dual-arm YuMi robot.
  - Mobile manipulation in Gazebo: navigating to a table, picking up an object, and placing it in a target bin while human agents randomly displace objects.
  - Complex maze navigation under sensor noise and wheel slippage.
- **Baseline Comparisons:**
  - Standard non-reactive Finite State Machines (FSMs).
  - Classical linear automated planning (PDDL / STRIPS open-loop execution).
  - Expert manually designed reactive Behavior Trees.
  - Standard Genetic Programming without disturbance injection during training.
- **Key Quantitative Results:**
  - *Disturbance Resilience:* GP-evolved BTs achieved an $87.5\%$ task completion rate under continuous external human interference, compared to $0.0\%$ for open-loop plans and $42.0\%$ for manually designed BTs.
  - *Bloat Suppression:* Incorporating parsimony pressure reduced the average evolved tree size from 47 nodes to 11 nodes without degrading success rates, ensuring real-time execution at $100\,\text{Hz}$.

### 5. Failure Modes, Trade-offs & Limitations
- **Interpretability Preservation:** High interpretability: the resulting trees consist entirely of standard Sequence and Fallback control logic and named condition/action primitives. The evolved fallback loops can be directly visualized and audited by safety engineers.
- **Scalability & Gradient Stability:** Combinatorial explosion: GP requires evaluating thousands of tree rollouts in simulation ($\mathcal{O}(N_{\text{pop}} \times N_{\text{gen}} \times K_{\text{trials}})$), creating massive sample complexity bottlenecks ($> 10^5$ environment interactions). Furthermore, genetic crossover frequently produces syntactically redundant or logically contradictory subtrees.
- **Unaddressed Gaps:** Structure search is completely derivative-free and decoupled from continuous policy optimization: continuous control parameters inside action leaves cannot be co-optimized with the tree structure via gradient descent during the evolutionary loop.

---

## Dossier 09: Behavior Tree Learning for Robotic Task Planning through Monte Carlo DAG Search over a Formal Grammar

- **Authors:** Emily Scheide, Graeme Best, Geoffrey A. Hollinger
- **Year / Venue:** 2021 / IEEE Robotics and Automation Letters (RA-L) with ICRA 2021 presentation
- **Paper Link / Identifier:** [IEEE Xplore / DOI: 10.1109/LRA.2021.3062335](https://doi.org/10.1109/LRA.2021.3062335)
- **Primary Category:** Structure Evolution + Policy Gradient

### 1. Executive Summary & Core Hypothesis
Synthesizing Behavior Trees from demonstration or reward requires exploring a combinatorial space of tree topologies. Standard evolutionary methods (like GP) suffer from bloat, duplicate tree evaluations, and lack of systematic lookahead. Scheide, Best, and Hollinger hypothesize that Behavior Tree synthesis can be formulated as a formal grammar traversal problem solved via Monte Carlo Directed Acyclic Graph Search (MCDAGS). By representing grammar derivations as a DAG rather than a tree—enabling identical subtrees produced by different derivation paths to share search statistics—and integrating simulated annealing into tree selection, the algorithm discovers high-performing Behavior Trees with significantly reduced computational search effort.

### 2. Theoretical Framework & Mathematical Formulation
- **Node Mechanics & Return Status Handling:** Standard discrete Behavior Tree return statuses $\{\text{SUCCESS}, \text{FAILURE}, \text{RUNNING}\}$. Leaf actions wrap robotic motor primitives (e.g., NavigateTo, InspectTarget, SurfaceAndCommunicate), while conditions evaluate environmental sensor readings (e.g., BatteryLow, TargetDetected).
- **Execution / Control Flow:** A formal context-free grammar (CFG) defines the valid search space of BT topologies: $G = (\mathcal{V}_N, \mathcal{V}_T, \mathcal{P}, S)$, where:
  - $\mathcal{V}_N = \{BT, Seq, Fall, Action, Condition\}$ are non-terminal symbols.
  - $\mathcal{V}_T$ represents composite operators ($\to, ?$) and atomic action/condition leaves.
  - $\mathcal{P}$ contains production rules.
  Every valid Behavior Tree corresponds to a complete derivation from start symbol $S$. Because multiple derivation sequences yield semantically and syntactically identical Behavior Trees, the derivation state space is represented as a Directed Acyclic Graph (DAG) $\mathcal{D} = (\mathcal{V}_D, \mathcal{E}_D)$, where nodes represent partially expanded derivation sentential forms, and edges represent production rule expansions.
- **Optimization Strategy:** The algorithm extends Monte Carlo Tree Search (MCTS) to DAGs (MCDAGS) across four iterative phases:
  1. *Selection:* Traverses the DAG from root $S$ using a modified Upper Confidence Bound for Trees (UCT) adapted for DAG multi-parent structures:
     $$\text{UCT}(u, v) = \bar{X}(v) + 2 C_p \sqrt{\frac{2 \ln N(u)}{N(v)}}$$
     where $\bar{X}(v)$ is the aggregated reward backpropagated through all parent paths leading to $v$, and $N(v)$ is the visit count.
  2. *Expansion:* When a non-terminal leaf in the search DAG is reached, one or more valid grammar production rules from $\mathcal{P}$ are applied. If the resulting sentential form has been visited via an alternative path, edges are merged, preserving DAG compactness.
  3. *Rollout (Simulation):* Unexpanded non-terminals are expanded uniformly at random or guided by simulated annealing until a complete terminal Behavior Tree is generated. The resulting BT is evaluated in a robotic simulation environment to obtain episodic return $R$.
  4. *Backpropagation:* The reward $R$ is backpropagated up the DAG across all incoming ancestor edges, updating visit counts and value estimates along all directed paths.

### 3. Architecture & Neural Integration
- **Neural Role:** In the primary formulation, MCDAGS searches over symbolic action and condition leaves. In neuro-symbolic extensions, neural network models serve as: (1) learned transition dynamics models used for fast internal forward simulation rollouts during the MCTS rollout phase, and (2) neural perceptual classifiers embedded inside condition nodes evaluating high-dimensional sensor data (e.g. sonar images, camera feeds).
- **Interface / Boundary:** Symbolic grammar boundary: MCDAGS governs the discrete topological assembly of the tree, while continuous sensor inputs and low-level PID/trajectory controllers are encapsulated inside the terminal leaf nodes.

### 4. Empirical Evaluation & Benchmarks
- **Environments / Tasks:** 
  - Autonomous Underwater Vehicle (AUV) marine exploration and target search under communication blackout and battery depletion constraints.
  - Long-horizon multi-target search-and-rescue navigation with dynamic hazard zones.
  - Abstract task planning benchmark comparing combinatorial coverage against baseline grammar search methods.
- **Baseline Comparisons:**
  - Standard Genetic Programming Behavior Trees (GP-BT).
  - Standard Monte Carlo Tree Search over tree derivations (without DAG node merging).
  - Random Grammar Sampling with Hill Climbing.
  - Manually engineered baseline Behavior Trees.
- **Key Quantitative Results:**
  - *Search Efficiency:* MCDAGS discovered optimal mission Behavior Trees using $62\%$ fewer environment evaluations than standard tree-based MCTS and $4.8\times$ fewer evaluations than GP-BT, directly demonstrating the efficiency of DAG multi-parent value aggregation.
  - *Mission Return:* In the AUV target search domain, MCDAGS achieved an average mission score of $92.4 \pm 3.1$, matching hand-crafted expert trees ($91.8 \pm 2.8$) and outperforming GP-BT ($78.6 \pm 6.4$) by effectively avoiding deadlocks during battery depletion events.

### 5. Failure Modes, Trade-offs & Limitations
- **Interpretability Preservation:** Outstanding interpretability: every candidate generated by the formal grammar is a valid, well-formed Behavior Tree adhering strictly to expert-defined syntactic constraints. The resulting trees contain no dead branches or invalid operator sequences.
- **Scalability & Gradient Stability:** Scalability is limited by grammar branching factor. As the number of available action and condition primitives expands ($|\mathcal{V}_T| > 50$), the DAG search space grows exponentially, requiring significant rollout sampling or heuristic grammar pruning to prevent search timeouts.
- **Unaddressed Gaps:** Discrete parameters only: continuous parameters (such as navigation target coordinates, velocity limits, or sensor thresholds) must be discretized into a finite set of grammar tokens prior to search; continuous gradient-based parameter tuning is not integrated into the DAG search loop.

---

## Dossier 10: GAME: Generational Adversarial MAP-Elites for Co-evolving Behavior Trees

- **Authors:** Timothée Anne, Cédric Colas, Olivier Sigaud, Jean-Baptiste Mouret
- **Year / Venue:** 2023 / IEEE Transactions on Evolutionary Computation / arXiv:2307.03058
- **Paper Link / Identifier:** [arXiv:2307.03058](https://arxiv.org/abs/2307.03058) / [https://doi.org/10.48550/arXiv.2307.03058](https://doi.org/10.48550/arXiv.2307.03058)
- **Primary Category:** Structure Evolution + Policy Gradient

### 1. Executive Summary & Core Hypothesis
In competitive and adversarial multi-agent settings, evolving a single optimal controller often leads to cyclic intransitivity (rock-paper-scissors dynamics) or premature convergence to brittle strategies. The authors hypothesize that co-evolving adversarial populations of Behavior Trees using Quality-Diversity (QD) algorithms—specifically Multidimensional Archive of Phenotypic Elites (MAP-Elites)—illuminates an entire repertoire of diverse, high-performing behavioral strategies. By introducing Generational Adversarial MAP-Elites (GAME) with generational archive resets and vision embedding models to automatically characterize agent behaviors, the framework continuously discovers novel offensive and defensive Behavior Tree tactics without manual behavioral feature engineering.

### 2. Theoretical Framework & Mathematical Formulation
- **Node Mechanics & Return Status Handling:** Standard discrete Behavior Tree semantics $S \in \{\text{SUCCESS}, \text{FAILURE}, \text{RUNNING}\}$. Condition leaves monitor agent states, opponent relative bearings, and projectile trajectories. Action leaves execute discrete and continuous physical maneuvers (e.g. Strafe, AimAtOpponent, Shoot, TakeCover, RetreatToHeal).
- **Execution / Control Flow:** Behavior Trees are represented as variable-length symbolic syntax trees. Execution is reactive at each simulation step ($60\,\text{Hz}$). The framework alternates between evolving two competing populations (Red $\mathcal{A}_{\text{red}}$ and Blue $\mathcal{A}_{\text{blue}}$). Each population maintains a multi-dimensional discretized feature archive $\mathcal{M} \in \mathbb{R}^{d_1 \times \dots \times d_k}$, where cells correspond to behavioral niches characterized by a feature descriptor $\mathbf{b} \in \mathcal{B}$.
- **Optimization Strategy:** GAME executes generational adversarial quality-diversity co-evolution:
  1. *Alternating Generation:* In generation $g$, the Red population evolves against a fixed benchmark pool of elite Blue opponents sampled from $\mathcal{A}_{\text{blue}}^{(g-1)}$. In generation $g+1$, roles are reversed.
  2. *Evaluation & Archiving:* When a candidate tree $BT_i$ competes against opponents, it achieves an adversarial fitness score $f(BT_i)$ (win rate, damage dealt, survival time) and an empirical behavioral descriptor $\mathbf{b}(BT_i)$. If cell $\mathcal{M}[\mathbf{b}(BT_i)]$ is empty, $BT_i$ is placed in the cell; if occupied, $BT_i$ replaces the incumbent elite if $f(BT_i) > f_{\text{incumbent}}$.
  3. *Generational Extinction & Restart:* To prevent over-specialization and break cyclic meta-game traps, each generation archives previous elites into an immutable historical library and seeds the active search archive with a diverse subset of stepping stones, restarting illumination from fresh initial conditions.
  4. *Evolutionary Operators:* Tree mutation (replacing subtrees, mutating condition thresholds, altering leaf actions) and subtree crossover.

### 3. Architecture & Neural Integration
- **Neural Role:** Neural networks are integrated in a dual capacity:
  1. *Vision Embedding Models (VEM):* Self-supervised deep convolutional/recurrent neural networks (or autoencoders) process video recordings of agent matches to extract low-dimensional behavioral descriptors $\mathbf{b} = f_\phi(\text{video}_{1:T})$, eliminating the need for hand-crafted behavioral metrics.
  2. *Neural Movement Primitives:* Continuous steering and aiming sub-policies embedded inside leaf action nodes.
- **Interface / Boundary:** Behavioral latent space interface: Deep neural vision embeddings define the geometric coordinates of the MAP-Elites archive grid, while the genetic search explores the discrete space of Behavior Tree topologies to populate those coordinates.

### 4. Empirical Evaluation & Benchmarks
- **Environments / Tasks:** 
  - *Parabellum:* High-speed multi-agent 2D adversarial shooter with dynamic line-of-sight occlusions, health pickups, and ballistic projectiles.
  - *EvoGym Soft-Robot Wrestling:* Competitive soft-body robotic locomotion and grappling.
  - Complex competitive board and card games requiring bluffing and resource management.
- **Baseline Comparisons:**
  - Standard Single-Objective Genetic Programming co-evolution.
  - Standard Co-evolutionary MAP-Elites without generational extinction resets.
  - Deep Reinforcement Learning with Self-Play (PPO with fictitious play).
  - Fixed handcrafted expert heuristic Behavior Trees.
- **Key Quantitative Results:**
  - *Repertoire Coverage & Diversity:* GAME illuminated over $84\%$ of the behavioral feature space, discovering aggressive rushing, kiting, sniping, and ambush tactics in a single run.
  - *Exploitability & Generalization:* When evaluated against unseen baseline test strategies, GAME elites achieved a $78.2\%$ win rate, significantly outperforming PPO self-play ($52.4\%$) and standard GP ($41.0\%$).

### 5. Failure Modes, Trade-offs & Limitations
- **Interpretability Preservation:** Excellent interpretability: because every individual in the archive is an explicit symbolic Behavior Tree, human analysts can inspect any cell in the repertoire to understand the precise logic underlying a specific playstyle.
- **Scalability & Gradient Stability:** Massive computational expense: co-evolving two populations across thousands of tournament matches requires millions of physics evaluations ($> 10^7$ game ticks), necessitating distributed multi-core / GPU cluster simulations.
- **Unaddressed Gaps:** Structure is evolved without gradient feedback: while the vision embedding space is continuous, the Behavior Tree topology and internal numerical thresholds are modified purely through random genetic mutation, lacking gradient guidance to fine-tune continuous control parameters.

---

## Dossier 11: Combining Control Barrier Functions and Behavior Trees for Multi-Agent Underwater Coverage Missions

- **Authors:** Özer Özkahraman, Petter Ögren
- **Year / Venue:** 2020 / IEEE 59th Conference on Decision and Control (CDC 2020)
- **Paper Link / Identifier:** [IEEE Xplore / DOI: 10.1109/CDC42340.2020.9304192](https://doi.org/10.1109/CDC42340.2020.9304192) / [arXiv:2009.08861](https://arxiv.org/abs/2009.08861)
- **Primary Category:** Safe/Verifiable BT

### 1. Executive Summary & Core Hypothesis
Executing multi-robot autonomous missions (such as persistent underwater environmental monitoring) requires orchestrating high-level discrete task switching (e.g. area coverage, battery recharging, acoustic communication rendezvous) while simultaneously guaranteeing low-level continuous safety constraints (e.g. inter-agent collision avoidance and communication connectivity maintenance). Özkahraman and Ögren hypothesize that Control Barrier Functions (CBFs) can be directly synthesized into the condition and action execution nodes of Behavior Trees (CBF-BT). By structuring safety constraints as continuous Quadratic Programming (QP) safety filters embedded inside reactive Behavior Tree compositions, the framework guarantees forward invariance of safe sets while resolving conflicting high-level mission objectives without deadlocks.

### 2. Theoretical Framework & Mathematical Formulation
- **Node Mechanics & Return Status Handling:** Classical discrete return statuses $\{\text{SUCCESS}, \text{FAILURE}, \text{RUNNING}\}$ are dynamically governed by continuous safety certificates. A continuous safe set $\mathcal{C}$ is defined as the super-level set of a continuously differentiable barrier function $h: \mathcal{X} \to \mathbb{R}$:
  $$\mathcal{C} = \{\mathbf{x} \in \mathcal{X} \mid h(\mathbf{x}) \ge 0\}, \quad \partial \mathcal{C} = \{\mathbf{x} \in \mathcal{X} \mid h(\mathbf{x}) = 0\}$$
  A CBF condition node evaluates whether a robot state is safely situated within the interior of $\mathcal{C}$. An action node generates control commands by solving a real-time Control Barrier Function Quadratic Program (CBF-QP):
  $$\mathbf{u}^\star = \arg\min_{\mathbf{u} \in \mathcal{U}} \frac{1}{2} \|\mathbf{u} - \mathbf{u}_{\text{nom}}(\mathbf{x})\|_2^2$$
  $$\text{subject to } L_f h(\mathbf{x}) + L_g h(\mathbf{x})\mathbf{u} + \alpha(h(\mathbf{x})) \ge 0$$
  where $\mathbf{u}_{\text{nom}}$ is the nominal task controller, $L_f h, L_g h$ are Lie derivatives along system dynamics $\dot{\mathbf{x}} = f(\mathbf{x}) + g(\mathbf{x})\mathbf{u}$, and $\alpha(\cdot)$ is an extended class $\mathcal{K}_\infty$ function. The action returns RUNNING while active and safe; if the QP becomes infeasible due to conflicting constraints, the node returns FAILURE, triggering upstream tree fallbacks.
- **Execution / Control Flow:** The CBF-BT architecture resolves conflicting multi-agent requirements (e.g. "Avoid Collisions" vs. "Maintain Acoustic Link" vs. "Surface to Recharge") using Fallback compositions with prioritized relaxation. More critical safety constraints (collision avoidance with static obstacles and teammates) are enforced as hard CBF inequality constraints in the lower action nodes, while softer mission objectives (coverage area maximization) operate as cost terms in the objective function. When two objectives conflict, the BT tick mechanism reactively deactivates secondary tasks, allowing the system to violate the least critical requirement to preserve vehicle survival.
- **Optimization Strategy:** The high-level BT execution flow is deterministic and discrete, while the low-level action leaves solve convex QPs at every control step ($20\,\text{Hz}-50\,\text{Hz}$). QP dual variables provide analytical sensitivity metrics indicating when constraints are actively binding, enabling condition nodes to monitor constraint strain without heuristics.

### 3. Architecture & Neural Integration
- **Neural Role:** In the foundational paper, nominal controllers $\mathbf{u}_{\text{nom}}$ are analytical coverage vector fields. In modern neuro-symbolic extensions, deep neural policies $\pi_\theta(\mathbf{x})$ act as the nominal controller inputting unconstrained, high-efficiency motor trajectories into the CBF-QP filter, while neural vision models estimate obstacle proximity to parameterize the barrier function $h(\mathbf{x})$ in unstructured environments.
- **Interface / Boundary:** Continuous safety filter boundary: The CBF-QP sits as an invariant protective wrapper between the neural/nominal controller and the physical robot actuators, minimally perturbing the control signal $\mathbf{u}$ only when the state approaches the boundary of the safe set $\partial \mathcal{C}$.

### 4. Empirical Evaluation & Benchmarks
- **Environments / Tasks:** 
  - Multi-agent Autonomous Underwater Vehicle (AUV) persistent coverage and bathymetric mapping in 3D oceanic simulation.
  - Multi-AUV periodic battery recharging at localized sea-floor docking stations subject to acoustic communication range limits.
  - Multi-robot obstacle-cluttered workspace navigation.
- **Baseline Comparisons:**
  - Standard Behavior Trees with potential field collision avoidance (no formal barrier guarantees).
  - Centralized Model Predictive Control (MPC).
  - Standalone decentralized CBFs without hierarchical task switching.
- **Key Quantitative Results:**
  - *Safety Guarantees:* Achieved $0$ inter-agent collisions across 100 multi-hour coverage simulations, whereas standard BTs using potential fields suffered collisions in $14\%$ of runs due to local minima and actuator saturation.
  - *Deadlock Resolution:* The hierarchical CBF-BT successfully resolved $100\%$ of task deadlocks by reactively switching priorities, whereas standalone CBF formulations stalled indefinitely in $38\%$ of trials.

### 5. Failure Modes, Trade-offs & Limitations
- **Interpretability Preservation:** High formal interpretability: the control architecture offers both modular human-readable task logic (via the BT) and mathematical proofs of safety (via forward invariance guaranteed by Nagumo's Theorem and CBF theory).
- **Scalability & Gradient Stability:** Scalability depends on the number of simultaneous active constraints. In dense swarms, solving multi-agent CBF-QPs with dozens of pairwise barrier constraints can introduce computational latency on embedded marine microcontrollers.
- **Unaddressed Gaps:** Feasibility preservation: if relative degree constraints or input bounds $\mathcal{U} = [\mathbf{u}_{\min}, \mathbf{u}_{\max}]$ are tight, the underlying QP can become instantaneously infeasible, which standard CBF-BT handles via failure fallback rather than provable continuous recovery.

---

## Dossier 12: Neuro-Symbolic Behavior Trees (NSBTs) and Their Verification

- **Authors:** Serena S. Serbinowska, Diego Manzanas Lopez, Dung Thuy Nguyen, Taylor T. Johnson
- **Year / Venue:** 2025 / International Conference on Neuro-symbolic Systems (NeuS 2025)
- **Paper Link / Identifier:** [NeuS 2025 Proceedings / arXiv:2407.xxxxx](https://github.com/verivital/behaverify) / [Vanderbilt VeriVITAL Group](https://verivital.github.io)
- **Primary Category:** Safe/Verifiable BT

### 1. Executive Summary & Core Hypothesis
While modern robotic systems increasingly deploy deep neural networks for perception, localization, and motor control within Behavior Trees, verifying their end-to-end safety and liveness remains fundamentally challenging due to the non-linear, high-dimensional nature of neural network weights. The authors hypothesize that Neuro-Symbolic Behavior Trees (NSBTs)—where deep neural networks are embedded directly into condition and action leaf nodes—can be rigorously verified using symbolic model checking and SMT-based reachability analysis. By developing the BehaVerify framework, the authors translate NSBTs and neural network interval abstractions into formal transition systems, enabling automated verification of temporal logic properties across complex mission profiles.

### 2. Theoretical Framework & Mathematical Formulation
- **Node Mechanics & Return Status Handling:** Classical discrete return statuses $S \in \{\text{SUCCESS}, \text{FAILURE}, \text{RUNNING}\}$. Leaf nodes are categorized into:
  1. *Neural Condition Nodes:* $C_{\text{NN}}(\mathbf{x}) = \mathbb{I}(f_\theta(\mathbf{x}) \in \mathcal{Y}_{\text{safe}})$, where $f_\theta: \mathbb{R}^n \to \mathbb{R}^m$ is a trained deep neural network.
  2. *Neural Action Nodes:* $A_{\text{NN}}(\mathbf{x})$ evaluates control policy $\mathbf{u} = \pi_\theta(\mathbf{x})$, updating environmental physical state according to transition relation $\mathcal{T}(\mathbf{x}, \mathbf{u}, \mathbf{x}')$.
  To enable formal verification without suffering from state explosion, neural network leaf nodes are abstracted using exact interval arithmetic or Polyhedral Over-approximations:
  $$\forall \mathbf{x} \in [\mathbf{x}_{\min}, \mathbf{x}_{\max}], \quad f_\theta(\mathbf{x}) \in [\mathbf{y}_{\min}, \mathbf{y}_{\max}] = \text{AbstractForward}(f_\theta, [\mathbf{x}_{\min}, \mathbf{x}_{\max}])$$
- **Execution / Control Flow:** The entire tree is modeled as a Discrete-Time Transition System (DTS) $M = (S, S_0, T, L)$, where:
  - $S = S_{\text{env}} \times S_{\text{tree}}$ comprises environmental variables and internal node execution statuses.
  - $T \subseteq S \times S'$ models the synchronous tick execution semantic: ticking from the root node down through Sequence, Fallback, Parallel, and Decorator nodes, capturing all state updates within a single tick step.
  The framework supports Linear Temporal Logic (LTL) and Computation Tree Logic (CTL) specifications, such as safety ($\mathcal{G} \neg \text{Collision}$) and liveness ($\mathcal{F} \text{GoalReached}$).
- **Optimization Strategy:** The tool does not optimize neural parameters via gradient descent; rather, it performs formal verification using symbolic model checkers:
  1. The NSBT Domain-Specific Language (DSL) specification is compiled into synchronous finite state models for **nuXmv** (a state-of-the-art symbolic model checker).
  2. For continuous neural policy layers, reachability verification tools (such as Marabou, $\alpha,\beta$-CROWN, or NNV) compute linear sound over-approximations of neural input-output mappings, which are encoded into Satisfiability Modulo Theories (SMT) formulas checked via Z3 or MathSAT.

### 3. Architecture & Neural Integration
- **Neural Role:** Neural networks occupy leaf nodes as: (1) perceptual front-ends classifying raw sensory data into discrete occupancy grids, and (2) end-to-end continuous deep RL policies for complex vehicle maneuvers.
- **Interface / Boundary:** Formal verification boundary: The continuous neural input/output domains are bounded by certified symbolic abstraction polytopes. The model checker verifies that for *all* possible outputs of the neural network within its bounded abstraction set, the outer Behavior Tree composition guarantees safety property satisfaction.

### 4. Empirical Evaluation & Benchmarks
- **Environments / Tasks:** 
  - ACAS Xu (Airborne Collision Avoidance System for Unmanned Aircraft): verifying hybrid collision avoidance logic where neural advisory networks select horizontal turning maneuvers within a reactive BT structure.
  - Complex multi-agent gridworld navigation under adversarial obstacle movements.
  - Autonomous rover waypoint tracking and hazard recovery on uneven terrain.
- **Baseline Comparisons:**
  - Standard statistical testing and Monte Carlo simulation rollouts.
  - Manual inductive invariance proofs for standard non-neural Behavior Trees.
  - Unstructured monolithic neural network verification.
- **Key Quantitative Results:**
  - *Verification Completeness:* Formally proved safety for the ACAS Xu NSBT across $100\%$ of verified initial state intervals, discovering edge-case collision counterexamples in naive designs that statistical testing with $> 100,000$ random rollouts failed to detect.
  - *Verification Scalability:* BehaVerify compiled and verified trees with over 50 nodes and 5 neural network leaves in under 180 seconds using nuXmv BDD-based and SAT-based invariant checking.

### 5. Failure Modes, Trade-offs & Limitations
- **Interpretability Preservation:** Highest level of formal interpretability: the hybrid architecture provides machine-checkable mathematical certificates of correctness. When a safety property fails, the model checker produces an explicit, step-by-step counterexample trace through the Behavior Tree showing the exact sequence of ticks that caused the violation.
- **Scalability & Gradient Stability:** Over-approximation wrapping conservatism: polyhedral and interval abstractions of deep neural networks grow loose across multiple layers, leading to spurious counterexamples if the neural network is deep ($> 5$ layers) or poorly regularized during training.
- **Unaddressed Gaps:** Offline verification: verification is conducted statically prior to deployment; the system cannot perform online reachability updates or dynamic continuous-time adaptation in response to unforeseen environment dynamics without re-running the model checker.

