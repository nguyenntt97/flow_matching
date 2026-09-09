# GAME: Generational Adversarial MAP-Elites for Co-evolving Behavior Trees

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
  - *Exploitability & Generalization:* When evaluated against unseen baseline test strategies, GAME elites achieved a $78.2\%$ win rate, significantly outperforming PPO self-play ($52.4\%$, which collapsed into a single narrow counter-strategy) and standard GP ($41.0\%$).

### 5. Failure Modes, Trade-offs & Limitations
- **Interpretability Preservation:** Excellent interpretability: because every individual in the archive is an explicit symbolic Behavior Tree, human analysts can inspect any cell in the repertoire to understand the precise logic underlying a specific playstyle (e.g., inspecting the tree for the "ambush from cover" niche).
- **Scalability & Gradient Stability:** Massive computational expense: co-evolving two populations across thousands of tournament matches requires millions of physics evaluations ($> 10^7$ game ticks), necessitating distributed multi-core / GPU cluster simulations.
- **Unaddressed Gaps:** Structure is evolved without gradient feedback: while the vision embedding space is continuous, the Behavior Tree topology and internal numerical thresholds are modified purely through random genetic mutation, lacking gradient guidance to fine-tune continuous control parameters.

