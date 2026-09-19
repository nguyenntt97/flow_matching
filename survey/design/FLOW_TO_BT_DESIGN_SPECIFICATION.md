# Flow-to-Behavior-Tree (Flow2BT): Design Specification & Brainstorming Compendium

**Document Status:** Architectural Proposal & Brainstorming Specification  
**Location:** `/home/nguyen/projects/flow_matching/survey/design/`  
**Theoretical Anchors:**  
- [arXiv:2605.00414](https://arxiv.org/abs/2605.00414) (*Trees to Flows and Back*, Ramachandran & Sra, 2026)  
- [Huang et al. (2025)](https://openreview.net/forum?id=NeuS2025_diff_bt) (*Differentiable Synthesis of Behavior Trees*)  
- [Özkahraman & Ögren (2020)](https://doi.org/10.1109/LRA.2020.2982352) (*CBF-BT*) & [Serbinowska et al. (2025)](https://verivital.github.io) (*BehaVerify*)  
- [Hémono et al. (2026/2027)](https://doi.org/10.1016/j.rcim.2026.103358) (*Automatic BT Generation Review*)  
- **Task Anchor:** [Bae et al. (2025)](https://arxiv.org/abs/2504.04756) (*Continuous Locomotive Crowd Behavior Generation - CrowdES*)

---

## 1. Executive Summary & Design Vision

### 1.1 The Core Motivation
Continuous generative policies—specifically **Flow Matching** and **Diffusion Policies**—have achieved state-of-the-art success in high-dimensional, multimodal robotic manipulation. However, deploying continuous neural vector fields $v_\theta(\mathbf{x}, t)$ directly in safety-critical robotics suffers from three fatal bottlenecks:
1. **Computational Latency:** Multiple ODE integration steps per control tick cap control frequency at $5 - 20\,\text{Hz}$, inadequate for stiff contact dynamics ($> 100 - 500\,\text{Hz}$).
2. **Black-Box Opacity:** Lacking symbolic logic, end-to-end continuous neural policies cannot be formally verified, audited, or certified.
3. **Fragility to Interruptions:** When an external disturbance occurs (e.g. object slippage), continuous flows lack explicit precondition/postcondition semantics and cannot execute reactive recovery loops.

Conversely, **Behavior Trees (BTs)** provide modularity, reactivity, human interpretability, and verifiable safety guarantees. However, learning BT topologies from scratch using combinatorial search (Genetic Programming) suffers from prohibitive sample complexity, while pure differentiable relaxations (Huang et al., 2025) suffer from severe *discretization-execution gaps*.

### 1.2 The Design Hypothesis
**We can systematically distill a continuous Flow Matching model into a certified, high-frequency, reactive Learnable Behavior Tree (LBT).** 

By leveraging Ramachandran & Sra's proof of **tree-flow duality** (continuous flows running in reverse time topologically generate hierarchical dendrograms), we use the continuous flow model as an exploratory *macro-teacher* to discover the natural task branching structure, condition boundaries, and motor primitives.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    CONTINUOUS TEACHER (Flow Matching Policy)                    │
│                 dx/dt = v_theta(x, t)  (Multimodal, High-Curvature)             │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                    Trajectory Rollouts  │  Reverse-Time Bifurcation
                    & Spectral Distance  ▼  (Kramers-Moyal Coarse-Graining)
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     TOPOLOGICAL DISCOVERY (Dendrogram Extraction)               │
│                  T_topo : Discovers Decision Hierarchy & Subgoals                │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                   Support Vector /      │  Linear Dynamic Compression
                   Stump Induction       ▼  (DMPs / Ridge Regression)
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    REACTIVE BEHAVIOR TREE ASSEMBLY & SAFETY                      │
│        • Control Nodes: Sequence (->) & Fallback (?)                            │
│        • Condition Leaves: C_k(s) = I(w_k^T s + b_k >= 0)                       │
│        • Action Leaves: Fast Motor Primitives (DMPs / LQR) @ > 500 Hz           │
│        • Safety Guards: Real-Time CBF-QP Filters                                │
│        • Verification: Formal nuXmv / SMT Model Checking                        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Synthesis of Survey Findings Across the Four Pillars

The design synthesizes findings from all 13 papers in the survey:

| Pillar / Research Line | Key Mechanism Imported | Role in Flow2BT Architecture |
| :--- | :--- | :--- |
| **Pillar 1: Differentiable & Continuous Limits**<br>*(Ramachandran 2026, Huang 2025, Frosst 2017)* | • Reverse-time flow bifurcation (Kramers-Moyal)<br>• Continuous tri-state vectors $[p_s, p_f, p_r] \in \Delta^2$<br>• Hyperplane decision gating | Extracts the macro-tree topology without grammar search; enables local differentiable fine-tuning. |
| **Pillar 2: Neuro-Symbolic Leaf Primitives**<br>*(Chatzilygeroudis 2021, Pereira 2015, Sprague 2022)* | • Dynamical Movement Primitives (DMPs)<br>• SMDP Option mapping<br>• Guarded Fallbacks with ROA Invariants | Compresses unimodal continuous flow branches into fast, certified, parameter-efficient motor leaves. |
| **Pillar 3: Structure Learning & Industry 5.0**<br>*(Iovino 2021, Scheide 2021, Hémono 2026)* | • Parsimony pressure against tree bloat<br>• Multi-objective task/ergonomic metrics<br>• Collaborative safety-envelope preemption | Guides minimal tree construction; ensures human-robot collaborative safety and operator legibility. |
| **Pillar 4: Safe & Verifiable Architectures**<br>*(Özkahraman 2020, Serbinowska 2025)* | • Control Barrier Function (CBF-QP) filters<br>• BehaVerify DSL compiler & nuXmv verification | Provides formal forward invariance and machine-checkable safety certificates prior to physical robot deployment. |

---

## 3. Detailed Algorithmic Pipeline: The 5-Tier Architecture

```
                                  [ TIER 1: TEACHER ]
                            Continuous Flow Matching Model
                             v_theta(x, t),  t in [0, 1]
                                          │
                                          │ Rollout Trajectories
                                          ▼
                                 [ TIER 2: TOPOLOGY ]
                         Reverse-Time Trajectory Clustering
                      Pairwise Spectral Distance D(xi_i, xi_j)
                                          │
                                          │ Dendrogram Extraction
                                          ▼
                      [ TIER 3: CONDITIONS ]     [ TIER 4: ACTIONS ]
                      Bifurcation Hyperplanes    Unimodal Flow Compression
                       C_k(s) = I(w^T s + b)     DMP Goal & Shape Regression
                                 │                            │
                                 └────────────┬───────────────┘
                                              ▼
                                   [ TIER 5: SAFETY & BT ]
                                Assembled Reactive Behavior Tree
                                  Sequence (->) & Fallback (?)
                                 CBF-QP Forward Invariance Guard
                                              │
                                              ▼
                                 [ TIER 6: VERIFICATION ]
                               BehaVerify DSL -> nuXmv SMT
```

### Tier 1: Continuous Teacher Flow Generation
1. Train a Flow Matching model $v_\theta(\mathbf{x}, t)$ using [`AffineProbPath`](file:///home/nguyen/projects/flow_matching/flow_matching/path/affine.py#L15-L261) and [`CondOTScheduler`](file:///home/nguyen/projects/flow_matching/flow_matching/path/scheduler/scheduler.py#L104) on expert demonstrations.
2. Roll out an ensemble of $M$ phase-space trajectories across diverse initial states $s_0 \sim \mathcal{S}_0$:
   $$\Xi = \{\xi_i(t)\}_{i=1}^M, \quad \dot{\xi}_i(t) = v_\theta(\xi_i(t), t, s_0)$$

### Tier 2: Topological Clustering & Dendrogram Extraction
1. Define the pairwise spectral trajectory distance:
   $$D(\xi_i, \xi_j) = \int_0^1 \|\xi_i(t) - \xi_j(t)\|_2^2 \, dt + \lambda_{\text{term}} \|\xi_i(1) - \xi_j(1)\|_2^2$$
2. Perform agglomerative hierarchical clustering in reverse time ($\tau = 1 - t$).
3. Identify bifurcation nodes where trajectory clusters split into distinct strategies (e.g., "Left Approach" vs. "Right Approach"). This defines the tree topology $\mathcal{T}_{\text{topo}}$.

### Tier 3: Condition Node Induction
1. At each internal bifurcation node $k$ with child branches $\mathcal{T}_{\text{left}}$ and $\mathcal{T}_{\text{right}}$, collect the initial state vectors $\mathbf{s}$ of trajectories assigned to each branch.
2. Train a linear support vector machine (SVM) or soft logistic stump:
   $$C_k(\mathbf{s}) = \mathbb{I}(\mathbf{w}_k^\top \phi(\mathbf{s}) + b_k \ge 0)$$
3. These hyperplanes form the **Condition Nodes** evaluated by BT ticks at runtime.

### Tier 4: Action Leaf Parameterization & Dynamic Compression
1. Within each terminal leaf partition $\ell$, the trajectories $\Xi_\ell$ are unimodal and low-curvature.
2. Fit a **Dynamical Movement Primitive (DMP)**:
   $$\tau \dot{\mathbf{v}} = K(\mathbf{g}_\ell - \mathbf{x}) - D\mathbf{v} + \mathbf{f}_\ell(s)$$
   where the attractor goal $\mathbf{g}_\ell$ is the cluster centroid at $t=1$, and non-linear shape weights $\mathbf{w}_{\text{DMP}}$ are fit via linear ridge regression.
3. This compresses a multi-million parameter neural network into a deterministic ODE executable at $> 500\,\text{Hz}$.

### Tier 5: Reactive Behavior Tree Assembly & Safety Certification
1. **Control Node Mapping:**
   - Divergent strategy branches map to **Fallback nodes ($?$)**: if the primary branch encounters an obstacle, the tick falls back to the alternate branch.
   - Sequential phases (e.g. Approach $\to$ Grasp $\to$ Transport) map to **Sequence nodes ($\to$)**.
2. **Control Barrier Function (CBF) Wrapper:**
   Wrap each DMP leaf with a real-time QP safety filter (Özkahraman & Ögren, 2020):
   $$\mathbf{u}^\star = \arg\min_{\mathbf{u}} \|\mathbf{u} - \mathbf{u}_{\text{DMP}}(\mathbf{s})\|^2 \quad \text{s.t.} \quad \nabla h(\mathbf{s})^\top \dot{\mathbf{s}}(\mathbf{u}) + \alpha(h(\mathbf{s})) \ge 0$$
3. **Formal Verification:**
   Compile the assembled tree into the BehaVerify DSL (Serbinowska et al., 2025) and verify collision-free temporal logic properties ($\mathcal{G} \neg \text{Collision}$) via nuXmv before deploying to hardware.

---

## 4. Key Design Advantages Over Prior Work

| Metric / Dimension | Teacher Flow Matching Policy | Differentiable BT (Huang 2025) | Proposed Flow2BT Distillation |
| :--- | :--- | :--- | :--- |
| **Control Frequency** | $5 - 20\,\text{Hz}$ (ODE latency) | $\approx 50\,\text{Hz}$ (Soft supernet) | **$> 500\,\text{Hz}$ (Deterministic DMPs)** |
| **Topology Origin** | Implicit / Continuous | Grammar search (Gumbel-Softmax) | **Data-driven bifurcation extraction** |
| **Discretization Gap** | N/A | High (Hardening degrades accuracy) | **Zero (DMPs fit directly to clusters)** |
| **Safety Guarantees** | None (Empirical black-box) | None (Unverified neural leaves) | **Certified (CBF-QP + nuXmv verification)** |
| **Disturbance Recovery** | Brittle (Open-loop continuation) | Brittle (Chimera leaf activation) | **Instant Reactive Fallback ticks** |
| **Human Interpretability** | Zero (Opaque latent field) | Partial (Diffused during training) | **Full (Explicit Sequence/Fallback tree)** |

---

---

## 5. Main Task Mapping: Pedestrian Crowd Locomotion (Bae et al., 2025)

The theoretical framework directly addresses the challenges identified in **Bae et al. (CrowdES, arXiv:2504.04756)** for continuous pedestrian locomotion:

### 5.1 Verification of Theoretical Assumptions in the Pedestrian Domain
1. **Validation of the Need for Discrete Modes:**  
   Bae et al. found that pure continuous flow/diffusion models cannot generate realistic long-term locomotion because pedestrian behaviors are intrinsically hybrid: walking briskly, slowing down for group alignment, swerving to avoid oncoming walkers, and stopping to chat. They had to inject an ad-hoc Switching Dynamical System (SDS) with $B=8$ discrete states. In our Flow2BT framework, these 8 states naturally correspond to **unimodal leaf motor primitives** under a Behavior Tree.
2. **Failure of Markov Transitions vs. Need for Reactive Behavior Trees:**  
   In CrowdES, transitions between behavioral modes are sampled stochastically from a Markov chain ($P(b_f \mid b_h, \dots)$). In dense crowds, this leads to erratic switching, deadlocks in narrow bottlenecks, and collision failures. A Behavior Tree replaces stochastic Markov jumping with **deterministic Sequence preconditions** (`IsPathClear`) and **Fallback evasions** (`IfCollisionImminent -> EvadeRight -> Yield`).
3. **Closing the Safety Gap via CBF Invariance:**  
   CrowdES relies on data-driven social pooling, which occasionally allows inter-pedestrian overlaps ($C_R > 0$). In our BT, wrapping the pedestrian DMP leaves with Control Barrier Functions guarantees hard pairwise safety envelopes ($d(p_i, p_j) \ge d_{\text{min}}$).
4. **Computational Acceleration for Real-Time Simulation:**  
   Evaluating 50-step diffusion and recurrent transformers for $N=100$ agents is computationally prohibitive. Compressing each pedestrian's locomotive mode into closed-form DMPs allows simulating dense crowds at $>500\,\text{Hz}$.

```
                      Fallback (?) : "Pedestrian Navigation"
                            │
         ┌──────────────────┴──────────────────┐
         ▼                                     ▼
   Sequence (->) : "Nominal Walk"         Sequence (->) : "Collision Avoidance"
         │                                     │
    ┌────┴────────────┐                   ┌────┴────────────┐
    ▼                 ▼                   ▼                 ▼
Condition:         Action:            Condition:         Action:
ClearPath?      FollowNavMeshDMP   PedestrianImpending? EvadeRightDMP (CBF Guarded)
```

---

## 6. Implementation Roadmap for the Codebase

- [ ] **Module 1 (`flow2bt/rollout.py`):** Trajectory ensemble generator integrating [`ODESolver`](file:///home/nguyen/projects/flow_matching/flow_matching/solver/ode_solver.py#L17-L204) across initial condition grids.
- [ ] **Module 2 (`flow2bt/clustering.py`):** Spectral trajectory metric and reverse-time hierarchical dendrogram induction.
- [ ] **Module 3 (`flow2bt/conditions.py`):** Differentiable hyperplane and logistic condition induction at bifurcation nodes.
- [ ] **Module 4 (`flow2bt/primitives.py`):** Ridge regression DMP parameterizer fitting attractor goals and forcing functions.
- [ ] **Module 5 (`flow2bt/assembly.py`):** Assembly into standard BT formats (py_trees / XML) with embedded CBF-QP safety wrappers.
- [ ] **Module 6 (`flow2bt/verification.py`):** Export to BehaVerify DSL for symbolic nuXmv model checking.

