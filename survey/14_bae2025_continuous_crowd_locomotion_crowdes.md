# Continuous Locomotive Crowd Behavior Generation (CrowdES)

- **Authors:** Inhwan Bae, Junoh Lee, Hae-Gon Jeon
- **Year / Venue:** 2025 / arXiv preprint (arXiv:2504.04756 [cs.CV, cs.LG, cs.RO])
- **Paper Link / Identifier:** [arXiv:2504.04756](https://arxiv.org/abs/2504.04756) / [Project Website](https://ihbae.com/publication/crowdes/) / [Code: GitHub InhwanBae/CrowdES](https://github.com/InhwanBae/CrowdES)
- **Primary Category:** Task Anchor: Continuous Pedestrian Locomotion Simulator (Excluding Emitter; Focusing on NavMesh Guidance, Switching Dynamical Systems, and Footstep Trajectory Synthesis)

### 1. Executive Summary & Scope Focus (Locomotion Simulator)
While the overall CrowdES system includes a diffusion population emitter, our primary interest is strictly **Section 3.3: The Crowd Locomotion Simulator**. In this subsystem, an active pedestrian agent at position $\mathbf{c}_t = (x, y)$ must autonomously navigate toward a long-term destination $\mathbf{c}_{T_d}$ across complex, non-convex walkable terrain while avoiding collisions and interacting with dynamic neighboring pedestrians. The authors demonstrate that pure continuous trajectory regression is incapable of realistic crowd locomotion over long horizons. Instead, they propose a **Switching Dynamical System (SDS)**: decomposing pedestrian continuous locomotion into a hybrid architecture combining global Navigation Mesh (NavMesh) polyline waypoints, $B=8$ discrete data-driven behavioral modes, and a local recurrent footstep predictor.

### 2. Theoretical Framework & Mathematical Formulation
- **Problem Formulation:** Given a single scene image $\mathcal{I}$, generate a crowd scenario $\mathcal{V}$ containing $N$ agents over an extended horizon $T_{\mathcal{V}}$. Each agent $A = \{\kappa, \mathcal{T}\}$ has an agent category $\kappa$ and continuous trajectory $\mathcal{T} = [\mathbf{c}_{T_s}, \dots, \mathbf{c}_{T_d}]$, where $\mathbf{c}_t = (x, y) \in \mathbb{R}^2$, and $T_s, T_d$ are arrival and departure times.
- **Scene Layout & Terrain Parsing:** Using Grounded-SAM and SegFormer, the model extracts four structural maps:
  1. Semantic segmentation map $\mathcal{M}_S$ (buildings, roads, sidewalks, vegetation).
  2. Appearance map $\mathcal{M}_A$ (entry/exit gates where people emerge).
  3. Population density map $\mathcal{M}_P$ and discrete population count distribution $\mathcal{P}$.
  4. Binary traversable map $\mathcal{M}_W$ converted into a navigation mesh (NavMesh via Recast) for polyline waypoint search.
- **Locomotion Simulator Formulation (Switching Dynamical System):**
  Once an agent is placed in the scene, its task is to generate continuous footsteps towards $\mathbf{c}_{T_d}$:
  1. *Navigation Mesh Pathfinding:* Recast builds a traversable polygon mesh $\mathcal{M}_W$. An A* search yields a polyline to $\mathbf{c}_{T_d}$, from which a local waypoint $\mathbf{c}_{t, \text{nav}}$ is extracted at distance $\nu$ (walking pace) ahead of the current position $\mathbf{c}_t$.
  2. *Discrete Behavioral Mode Extraction ($B=8$ States):* The continuous trajectory is segmented into chunks of length $T_f = 20$ frames ($4\,\text{s}$). Normalizing these chunks across rotation, translation, and scale around $\mathbf{c}_{t, \text{nav}}$ and applying K-means clustering reveals $B = 8$ discrete behavioral primitives (e.g., straight walking, swerving left, swerving right, stopping/yielding, acceleration).
  3. *State-Switching Markov Transitions:* At every interval $T_f$, transition probabilities are predicted by network $\mu_\phi$:
     $$b_f \sim P(b_f \mid b_h, \mathcal{T}_{t-T_h:t}, \mathcal{H}_{\text{neighbors}}, \mathbf{C}_s)$$
     conditioned on past motion, neighboring pedestrians $\mathcal{H}$, and local terrain $\mathbf{C}_s$ ($16 \times 16\,\text{m}$ cropped map).
  4. *Continuous Trajectory Prediction:* Conditioned on the discrete mode $b_f$ and NavMesh target $\mathbf{c}_{t, \text{nav}}$, a recurrent trajectory predictor $\mu_\varphi$ generates the continuous footstep coordinates:
     $$\mathcal{T}_{t:t+T_f} = \mu_\varphi(b_f, \mathbf{c}_{t, \text{nav}}, \mathbf{C}_s)$$

### 3. Architecture & Mechanics of the Locomotion Simulator
- **Locomotion Inputs:** Current coordinates $\mathbf{c}_t$, goal $\mathbf{c}_{T_d}$, walking pace $\nu$, historical trajectory $\mathcal{T}_{t-T_h:t}$, neighbor trajectories $\mathcal{H}_{\text{neighbors}}$, and local terrain map $\mathbf{C}_s$.
- **Locomotion Outputs:** Step-by-step continuous footstep sequence $[\mathbf{c}_t, \dots, \mathbf{c}_{t+T_f}]$ and the active behavioral state $b_f \in \{1, \dots, 8\}$.
- **Core Structural Trade-off:** The continuous trajectory predictor $\mu_\varphi$ relies on the discrete state $b_f$ to break trajectory multimodality. Without $b_f$, continuous regression suffers from mode averaging (e.g. attempting to walk through oncoming obstacles).

### 4. Empirical Evaluation & Benchmarks
- **Environments / Datasets:**
  - Standard pedestrian trajectory benchmarks: ETH, UCY (Zara01, Zara02, Univ), Stanford Drone Dataset (SDD), and GC (Grand Central Station).
  - Evaluated on long-term crowd video streams (up to 10 hours of continuous activity).
- **Evaluation Metrics:**
  - *Scene-Level Realism:* Population Distribution Difference (PDD), Density Distribution Error (DDE).
  - *Individual-Level Accuracy:* Average Displacement Error (ADE), Final Displacement Error (FDE), Collision Rate ($C_R$).
- **Key Quantitative Results:**
  - Successfully generates hours of continuous crowd behaviors without population collapse or empty-scene artifacts.
  - Generates realistic group flocking, bidirectional stream formation in hallways, and collision-free bottleneck traversal across complex real-world layouts.

### 5. Relevance & Assumptions Verification for Flow-to-Behavior-Tree Distillation
- **Direct Validation of Discrete-Continuous Hybrid Need:** Bae et al. demonstrate that pure continuous trajectory generation is inadequate for realistic pedestrian locomotion. To capture real-world locomotion, they were forced to introduce an explicit **Switching Dynamical System with $B=8$ discrete states** governing actions like stopping, yielding, and swerving.
- **The Remaining Behavior Tree Gap in CrowdES:**
  1. *Heuristic Markov Transitions vs. Deterministic Preconditions:* CrowdES samples behavioral states stochastically from a learned transition distribution $P(b_f \mid \cdot)$. It lacks explicit **Sequence preconditions** (e.g. `IfPathObstructed -> Yield`) and **Fallback recovery logic** (e.g. `AttemptStep -> OnCollisionTrip -> EvadeLeft`).
  2. *Safety & Collision Invariance:* CrowdES occasionally produces inter-agent penetrations and near-collisions because it lacks hard barrier guarantees. Wrapping the continuous locomotion field in **Control Barrier Functions (CBFs)** resolves this.
  3. *Inference Latency:* Denoising agents via 50-step diffusion and recurrent neural network forward passes for every pedestrian in a crowd of $N=100$ agents is computationally expensive. Distilling these unimodal switching states into **Dynamical Movement Primitives (DMPs)** organized under a **reactive Behavior Tree** reduces latency to sub-millisecond execution.

