# Combining Control Barrier Functions and Behavior Trees for Multi-Agent Underwater Coverage Missions

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
- **Execution / Control Flow:** The CBF-BT architecture resolves conflicting multi-agent requirements (e.g. "Avoid Collisions" vs. "Maintain Acoustic Link" vs. "Surface to Recharge") using Fallback compositions with prioritized relaxation. More critical safety constraints (collision avoidance with static obstacles and teammates) are enforced as hard CBF inequality constraints in the lower action nodes, while softer mission objectives (coverage area maximization) operate as cost terms in the objective function. When two objectives conflict, the BT tick mechanism reactively deactivates secondary tasks, allowing the system to violate the least critical requirement (e.g. temporarily breaking network topology) to preserve vehicle survival.
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
  - *Deadlock Resolution:* The hierarchical CBF-BT successfully resolved $100\%$ of task deadlocks (e.g., an AUV needing to recharge but blocked by an acoustic connectivity constraint) by reactively switching priorities, whereas standalone CBF formulations stalled indefinitely in $38\%$ of trials.

### 5. Failure Modes, Trade-offs & Limitations
- **Interpretability Preservation:** High formal interpretability: the control architecture offers both modular human-readable task logic (via the BT) and mathematical proofs of safety (via forward invariance guaranteed by Nagumo's Theorem and CBF theory).
- **Scalability & Gradient Stability:** Scalability depends on the number of simultaneous active constraints. In dense swarms, solving multi-agent CBF-QPs with dozens of pairwise barrier constraints can introduce computational latency on embedded marine microcontrollers.
- **Unaddressed Gaps:** Feasibility preservation: if relative degree constraints or input bounds $\mathcal{U} = [\mathbf{u}_{\min}, \mathbf{u}_{\max}]$ are tight, the underlying QP can become instantaneously infeasible (the empty safe set problem), which standard CBF-BT handles via failure fallback rather than provable continuous recovery.

