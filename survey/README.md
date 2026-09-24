# Systematic Literature Survey: Learnable & Neural Behavior Trees

**Author:** Elite Research Scientist in Neuro-Symbolic Robotics & Autonomous Decision Systems  
**Primary Anchor Paper:** [arXiv:2605.00414](https://arxiv.org/abs/2605.00414) — *Trees to Flows and Back: Unifying Decision Trees and Diffusion Models* (Sai Niranjan Ramachandran & Suvrit Sra, 2026)  
**Repository Location:** `/home/nguyen/projects/flow_matching/survey`

---

## 1. Executive Research Overview & Foundational Anchor

Behavior Trees (BTs) have emerged as the premier reactive execution and task-switching architecture across robotics, autonomous systems, and interactive agents due to their intrinsic modularity, reactivity, and human-interpretability. However, classical BTs suffer from a fundamental theoretical barrier: **execution ticks propagate non-differentiable categorical return signals ($\{\text{SUCCESS}, \text{FAILURE}, \text{RUNNING}\}$)**. Consequently, structural synthesis has historically been confined to combinatorial search algorithms (Genetic Programming, Grammatical Evolution, Monte Carlo Tree Search), while parameter optimization has relied on derivative-free black-box methods.

This survey anchors itself upon the groundbreaking mathematical paradigm presented in **arXiv:2605.00414 (Ramachandran & Sra, 2026)**: **"Trees to Flows and Back: Unifying Decision Trees and Diffusion Models"**. 

```
                               ┌────────────────────────────────────────────────────────┐
                               │                 arXiv:2605.00414                       │
                               │               Ramachandran & Sra                       │
                               │      (Tree-Flow Duality / Continuous Limits)          │
                               └────────────────────────┬───────────────────────────────┘
                                                        │
          ┌───────────────────────────┬─────────────────┴───────────┬───────────────────────────┐
          ▼                           ▼                             ▼                           ▼
┌──────────────────┐        ┌──────────────────┐          ┌──────────────────┐        ┌──────────────────┐
│     Pillar 1     │        │     Pillar 2     │          │     Pillar 3     │        │     Pillar 4     │
│ Differentiable   │        │  Neuro-Symbolic  │          │  Evolutionary &  │        │   Verifiable &   │
│      BTs         │        │    Hybrid BTs    │          │ Structure Search │        │  Safe Neural BTs │
├──────────────────┤        ├──────────────────┤          ├──────────────────┤        ├──────────────────┤
│• Huang (2025)    │        │• Sprague (2022)  │          │• Iovino (2021)   │        │• Özkahraman(2020)│
│• Kontschieder'15 │        │• Chatzilygeroudis│          │• Scheide (2021)  │        │• Serbinowska'25  │
│• Frosst & Hinton │        │• Pereira & Engel │          │• Anne (2023)     │        │                  │
└──────────────────┘        └──────────────────┘          └──────────────────┘        └──────────────────┘
```

### The Mathematical Bridge of arXiv:2605.00414
Ramachandran & Sra prove that the hierarchical spatial partitioning of a decision tree corresponds to an entropy-increasing, coarse-graining Markov chain of probability distributions $\{p(\mathbf{x}, k)\}_{k=0}^T$. Under dyadic temporal refinement ($\Delta t = 2^{-n} \to 0$), the Kramers-Moyal expansion truncates by Pawula's Theorem at second order, proving that the continuum limit of hierarchical trees converges to a drift-diffusion Stochastic Differential Equation (SDE) and a deterministic probability flow Ordinary Differential Equation (ODE):
$$\frac{d\mathbf{x}}{dt} = \mu(\mathbf{x}, t) - \frac{1}{2} \sigma^2(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$$
The authors reveal that functional gradient tree boosting is asymptotically optimal for **Global Trajectory Score Matching (GTSM)** in the space of SDE trajectories, providing two operational algorithms:
1. **TreeFlow:** Tree-conditioned continuous flow matching, directing probability flows along tree-structured hierarchical routing paths.
2. **DSM-Tree:** Hierarchical score matching that distills discrete tree decision logic directly into continuous neural networks.

Applied to Autonomous Decision Systems and Learnable Behavior Trees (LBTs), this duality resolves the long-standing dichotomy between **discrete symbolic control structures** and **continuous neural policies**. It provides a rigorous foundation for continuous relaxations of Sequence and Fallback nodes, bidirectional distillation between neural vector fields and symbolic trees, and gradient-based co-design of tree topology and low-level controllers.

---

## 2. Taxonomy Across the Four Methodological Pillars

### Pillar 1: Differentiable & Gradient-Optimized BTs
*Focus: Continuous relaxations of BT control nodes (Sequence, Fallback, Parallel), end-to-end backpropagation via Gumbel-Softmax, soft logic, and neural decision forests.*
- [01_ramachandran2026_trees_to_flows.md](./01_ramachandran2026_trees_to_flows.md): Mathematical foundations of tree-flow duality, continuous-time SDE limits, and Global Trajectory Score Matching.
- [02_huang2025_differentiable_bt_synthesis.md](./02_huang2025_differentiable_bt_synthesis.md): Continuous relaxation of formal grammar derivation graphs and tri-state execution status vectors via smooth t-norms and Gumbel-Softmax routing.
- [03_kontschieder2015_deep_neural_decision_forests.md](./03_kontschieder2015_deep_neural_decision_forests.md): Foundational antecedent unifying deep CNN feature representations with stochastic tree routing functions.
- [04_frosst2017_soft_decision_trees.md](./04_frosst2017_soft_decision_trees.md): Distilling continuous neural networks into soft, differentiable hierarchical decision trees.

### Pillar 2: Neuro-Symbolic BT Architectures
*Focus: Symbolic tree topologies with neural network leaf nodes (action policies and perceptual condition evaluators), HRL Options framework mapped to BT semantics, and guarded execution.*
- [05_sprague2022_neural_controllers_bt.md](./05_sprague2022_neural_controllers_bt.md): Fallback composition embedding black-box continuous deep RL controllers under formal region-of-attraction invariant sets.
- [06_chatzilygeroudis2021_bt_movement_skills.md](./06_chatzilygeroudis2021_bt_movement_skills.md): Parameterized Behavior Trees wrapping Dynamical Movement Primitives (DMPs) optimized via constrained policy search for physical robot assembly.
- [07_pereira2015_options_learning_nodes_bt.md](./07_pereira2015_options_learning_nodes_bt.md): Formal mapping establishing equivalence between Behavior Tree subtrees and SMDP Options, introducing adaptive Learning Nodes with internal Q-learning.

### Pillar 3: Evolutionary & Hybrid Structure Learning
*Focus: Genetic Programming (GP), Monte Carlo DAG Search (MCDAGS), and Quality-Diversity (MAP-Elites) searching for BT topology while optimizing execution primitives.*
- [08_iovino2021_gp_bt_unpredictable.md](./08_iovino2021_gp_bt_unpredictable.md): Genetic Programming evolving reactive Behavior Trees under stochastic environmental perturbations and parsimony pressure.
- [09_scheide2021_mcdags_grammar_bt.md](./09_scheide2021_mcdags_grammar_bt.md): Synthesizing Behavior Trees through Monte Carlo Directed Acyclic Graph Search over formal grammars with simulated annealing.
- [10_anne2023_game_map_elites_bt.md](./10_anne2023_game_map_elites_bt.md): Generational Adversarial MAP-Elites co-evolving diverse competitive Behavior Trees using deep vision embedding models.
- [13_hemono2026_automatic_bt_generation_hrc.md](./13_hemono2026_automatic_bt_generation_hrc.md): Systematic review of automatic BT generation across classical planning, evolutionary metaheuristics, and LLMs for collaborative task planning in Industry 5.0.

### Pillar 4: Verifiable & Safe Neural BTs
*Focus: Control Barrier Functions (CBFs), Quadratic Programming safety filters, and formal verification of neural Behavior Trees using symbolic model checkers.*
- [11_ozkahraman2020_cbf_bt.md](./11_ozkahraman2020_cbf_bt.md): Control Barrier Function Behavior Trees (CBF-BT) resolving conflicting multi-agent mission objectives while guaranteeing forward invariance.
- [12_serbinowska2025_nsbt_verification.md](./12_serbinowska2025_nsbt_verification.md): BehaVerify framework for formal verification of Neuro-Symbolic Behavior Trees with deep neural network leaves using nuXmv and SMT solvers.

### Pillar 5: Flow Matching & Continuous-Discrete Hybrid Trajectory Forecasting
*Focus: Conditional flow matching (CFM), optimal transport displacement paths, one-step distillation (IMLE), discrete goal point conditioning, and physical barrier guidance for agent-level trajectory prediction.*
- [15_fu2025_moflow_onestep_flow_matching.md](./15_fu2025_moflow_onestep_flow_matching.md): One-step flow matching for human trajectory forecasting via Best-of-K CFM loss and IMLE distillation (00	imes$ speedup, SOTA on ETH-UCY and SDD).
- [16_xing2025_goalflow_multimodal_trajectories.md](./16_xing2025_goalflow_multimodal_trajectories.md): Goal-driven flow matching resolving trajectory divergence by coupling a discrete goal vocabulary (endpoint clusters) with 1-step rectified flow planning (SOTA on Navsim).
- [17_yan2025_trajflow_motion_prediction.md](./17_yan2025_trajflow_motion_prediction.md): Multi-modal motion prediction generating parallel trajectory modes in a single pass with self-conditioning and Plackett-Luce ranking losses on WOMD.
- [18_ye2024_tcfm_trajectory_conditional_flow_matching.md](./18_ye2024_tcfm_trajectory_conditional_flow_matching.md): Trajectory Conditional Flow Matching unifying prediction and generation with 00	imes$ speedup and 35% higher accuracy over diffusion models.
- [19_zhu2025_motion_field_regularized_flow_matching.md](./19_zhu2025_motion_field_regularized_flow_matching.md): Regularizing flow matching with neural implicit motion fields and test-time Signed Distance Function (SDF) gradient guidance to guarantee collision-free pedestrian navigation.
- [20_tan2025_flow_planner_interactive_behavior.md](./20_tan2025_flow_planner_interactive_behavior.md): Interactive multi-agent planning via fine-grained trajectory tokenization and classifier-free guided flow matching on nuPlan.
- [21_mao2026_low_rank_spectral_flow_matching.md](./21_mao2026_low_rank_spectral_flow_matching.md): Low-Rank Spectral Flow Matching (LR-SFM) in truncated DCT space, proving human trajectory diversity is sparse and achieving SOTA on ETH-UCY with fewer function evaluations.

---

## 3. Comprehensive Comparative Matrix

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
| **Hémono et al. (2026/2027)**<br>*Auto BT Generation in Industry 5.0* | Survey / Automatic BT Synthesis | **Hybrid / Varies** (Planning, Evolutionary, LLMs) | Learned / Generated (PDDL, GP, LLM prompt-to-BT) | Collaborative action primitives, cobot commands, safety condition leaves | Human-robot collaborative assembly, JSSP/ALBP scheduling, ergonomics |
| **Bae et al. (2025)**<br>*CrowdES: Continuous Crowd Locomotion* | Task Anchor: Pedestrian Locomotion | **Partial** (Diffusion Emitter + Markov Chain SDS) | Learned (Diffusion denoising + SDS state switching) | 2D Footstep coordinates $\mathbf{c}_t$, NavMesh polyline guidance | Multi-agent crowd locomotion (ETH, UCY, SDD, Grand Central) |
| **Fu et al. (2025)**<br>*MoFlow (CVPR 2025)* | Flow Matching Trajectory Forecasting | **Yes** (Continuous CFM + 1-step IMLE) | Fixed / Distilled Student | Multi-agent continuous coordinate trajectories | Pedestrian benchmarks (ETH-UCY, SDD, NBA SportVU) |
| **Xing et al. (2025)**<br>*GoalFlow (CVPR 2025)* | Discrete-Continuous Hybrid Flow | **Yes** (1-step Rectified Flow) | Hybrid (Discrete Goal Vocab + Flow) | Goal-conditioned trajectory generator | Autonomous navigation & planning (Navsim, nuScenes) |
| **Yan et al. (2025)**<br>*TrajFlow (IROS 2025)* | Multi-Agent Flow Matching | **Yes** (Parallel hBcflow paths) | Fixed (Self-conditioned network) | Multi-modal ranked trajectory heads | Multi-agent motion forecasting (WOMD) |
| **Ye & Gombolay (2024)**<br>*T-CFM (IROS 2024)* | Foundational Trajectory CFM | **Yes** (Continuous ODE flow) | Fixed (Learned velocity field) | Continuous velocity vector fields | Multi-agent tracking, aircraft flight, long-horizon planning |
| **Zhu et al. (2025)**<br>*Motion-Reg Flow (AAAI 2025)* | Feasible Flow Matching | **Yes** (Continuous ODE + SDF gradient) | Fixed (Motion field regularized) | Obstacle-avoidant pedestrian paths | Complex human navigation (SDD, Edinburgh Forum) |
| **Tan et al. (2025)**<br>*Flow Planner (NeurIPS 2025)* | Interactive Guided Flow | **Yes** (CFG velocity blending) | Fixed (Tokenized interactive transformer) | Multi-agent interactive trajectories | Closed-loop autonomous driving (nuPlan) |
| **Mao et al. (2026)**<br>*LR-SFM (KDD 2026)* | Spectral Flow Matching | **Yes** (Continuous spectral ODE) | Fixed (DCT low-rank projection) | Truncated DCT frequency coefficients | Human trajectory prediction (ETH-UCY, SDD, NBA) |

---

## 4. Synthesis: Open Theoretical Frontiers in Continuous, Learnable Behavior Trees

The convergence of hierarchical decision architectures with continuous-time dynamical systems—crystallized by Ramachandran & Sra's (2026) proof of the equivalence between hierarchical tree partitioning and drift-diffusion processes—has opened profound theoretical frontiers at the intersection of control theory, generative modeling, and neuro-symbolic robotics. 

First, **resolving the discretization-execution gap in differentiable behavior trees** remains an urgent theoretical open problem. Existing differentiable relaxations (such as Huang et al., 2025) approximate discrete control nodes using soft continuous t-norms (product or Łukasiewicz logic) and Gumbel-Softmax reparameterizations during backpropagation. However, hardening this continuous supernet into a deterministic, tick-based Behavior Tree for physical deployment causes severe performance degradation, as the discrete tree cannot reproduce the fractional co-activation of parallel branches that the neural optimizer exploited. Applying the Fokker-Planck continuum limit reveals that discrete BT execution is an un-annealed, coarse-grained discretization of a continuous probability flow ODE. Developing exact, boundary-preserving flow-matching schemes that guarantee zero loss of reactivity upon hardening represents a paramount mathematical challenge.

### 4.2 Resolving the Flow Matching vs. Discrete State System (CrowdES) Dilemma in Agent Trajectory Prediction

A critical conceptual question in continuous crowd locomotion is the comparative trade-off between **discrete-state systems** (such as CrowdES's =8$ discrete locomotion modes governed by a Markov transition network) and **continuous flow matching models** (such as Conditional Flow Matching policies predicting velocity fields $\mathbf{v}_	heta$). At first glance, practitioners often observe that discrete-state systems appear more robust for long-horizon closed-loop simulation, leading to the perception that flow matching is sub-optimal for agent-level trajectory prediction.

However, a comprehensive investigation across peer-reviewed literature from **CVPR, NeurIPS, IROS, AAAI, and KDD (2024–2026)** reveals the precise mathematical reasons for this phenomenon and establishes how state-of-the-art flow matching architectures resolve it:

1. **The Root of the Dilemma: Why Vanilla Flow Matching Appears Sub-Optimal:**
   - *Trajectory Divergence over Long Horizons:* As demonstrated by **GoalFlow (Xing et al., CVPR 2025)**, unconstrained flow matching trained to map Gaussian noise directly to future paths suffers from severe spatial divergence: small deviations in early velocity integration cause paths to drift off drivable/walkable geometry, yielding high collision rates ($> 5\%$) in closed-loop navigation despite strong open-loop metrics.
   - *The Static Obstacle Blindness:* As proved by **Zhu et al. (AAAI 2025)**, standard regression losses do not penalize vector fields that pass through static walls. In their benchmarks, vanilla flow matching suffered an obstacle violation rate of 4.8\%$ (.5	imes$ higher than discrete NavMesh baselines), explaining why discrete polygon pathfinding (like CrowdES's Recast NavMesh) feels superior in complex architectural layouts.
   - *The Numerical ODE Latency Penalty:* Integrating continuous neural ODEs requires 5–20 evaluation steps per agent per frame ($> 20 - 50\,	ext{ms}$ on GPU), whereas discrete lookup or single-pass regression evaluates in $< 1\,	ext{ms}$.

2. **How Peer-Reviewed Literature Resolves the Dilemma:**
   - **Discrete Goal / Mode Anchoring (GoalFlow, CVPR 2025):** Rather than treating discrete states and flow matching as mutually exclusive, GoalFlow proves they are fundamentally complementary. By constructing a **discrete goal vocabulary** from clustered trajectory endpoints (analogous to CrowdES's discrete behavioral clusters) and conditioning rectified flow matching on the selected goal point, GoalFlow completely eliminates trajectory divergence, achieving state-of-the-art 0.3$ PDMS on Navsim in a **single step**.
   - **One-Step Distillation via IMLE (MoFlow, CVPR 2025):** MoFlow establishes that a continuous flow teacher trained on multi-modal human trajectories (ETH-UCY, SDD) can be distilled via Implicit Maximum Likelihood Estimation into a **1-step generator** running at **00	imes$ speedup** ($< 1\,	ext{ms}$ per scene), simultaneously outperforming CrowdES, diffusion models, and CVAEs on minADE/minFDE.
   - **Test-Time SDF Gradient Steering (Zhu et al., AAAI 2025):** By injecting Signed Distance Function (SDF) barrier gradients directly into the flow matching velocity field ($\dot{	au}_t = v_	heta - lpha 
abla \mathcal{U}_{	ext{obs}}$), obstacle violation drops from 4.8\%$ to **bash.2\%* without requiring model retraining.
   - **Low-Rank Spectral Frequency Compression (LR-SFM, KDD 2026):** By performing flow matching in truncated DCT space, LR-SFM proves human trajectory diversity is concentrated in 3–4 spectral modes, reducing integration dimensionality by 6\%$ while enforcing kinematic smoothness.
   - **Calibrated Multi-Mode Ranking (TrajFlow, IROS 2025):** Uses self-conditioning and Plackett-Luce ranking losses to generate parallel multi-modal trajectories in a single pass with calibrated confidence.

3. **Validation of the Flow2BT Pipeline:**
   These findings directly validate our Flow2BT framework: Subsystem 1 provides the discrete NavMesh goal anchor $\mathbf{c}_{t, 	ext{nav}}$ (mirroring GoalFlow), Subsystem 2 trains the multi-modal Flow Teacher (mirroring MoFlow/T-CFM), Subsystem 5 distills trajectories into low-rank DMPs (mirroring LR-SFM), Subsystem 6 enforces 00\,	ext{Hz}$ preemption, and Subsystem 7a guarantees collision invariance via CBF-QP filters (mirroring test-time barrier guidance).

Second, the **unification of Control Barrier Functions (CBFs) with continuous-time score fields within dynamic Behavior Trees** offers a transformative path for safety-critical learning. In current CBF-BT architectures (Özkahraman & Ögren, 2020), safety is enforced through local, instantaneous Quadratic Programs that assume known analytical control-affine dynamics and frequently suffer from infeasibility when conflicting objectives arise. By viewing the hierarchy as a continuous vector field $\frac{d\mathbf{x}}{dt} = \mu(\mathbf{x}, t) - \frac{1}{2} \sigma^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$, barrier certificates can be cast directly as functional gradient constraints on the score field itself. This would enable provably forward-invariant continuous-time score matching, ensuring that end-to-end neural policies distilled into or guided by trees cannot penetrate unsafe state-space manifolds even under extreme environmental uncertainty.

Third, **bidirectional neural-symbolic distillation under non-stationary reactive control loops** remains unformalized. While DSM-Tree (Ramachandran & Sra, 2026) achieves score distillation for static tabular distributions, robotic systems operate in closed-loop environments where actions alter underlying state transitions. Formulating a temporal, closed-loop extension of Global Trajectory Score Matching (GTSM) that accounts for Bellman state-occupancy shifts will enable seamless bidirectional transfer: compiling large black-box foundation policies (e.g., Vision-Language-Action models) into auditable, formally verifiable Behavior Trees, and conversely, diffusing human-engineered safety logic directly into agile, continuous neural controllers.

---

## 5. Directory Contents & Links to Individual Paper Dossiers

- [01_ramachandran2026_trees_to_flows.md](./01_ramachandran2026_trees_to_flows.md) — *Trees to Flows and Back: Unifying Decision Trees and Diffusion Models*
- [02_huang2025_differentiable_bt_synthesis.md](./02_huang2025_differentiable_bt_synthesis.md) — *Differentiable Synthesis of Behavior Tree Architectures and Execution Nodes*
- [03_kontschieder2015_deep_neural_decision_forests.md](./03_kontschieder2015_deep_neural_decision_forests.md) — *Deep Neural Decision Forests*
- [04_frosst2017_soft_decision_trees.md](./04_frosst2017_soft_decision_trees.md) — *Distilling a Neural Network Into a Soft Decision Tree*
- [05_sprague2022_neural_controllers_bt.md](./05_sprague2022_neural_controllers_bt.md) — *Adding Neural Network Controllers to Behavior Trees without Destroying Performance Guarantees*
- [06_chatzilygeroudis2021_bt_movement_skills.md](./06_chatzilygeroudis2021_bt_movement_skills.md) — *Learning of Parameters in Behavior Trees for Movement Skills*
- [07_pereira2015_options_learning_nodes_bt.md](./07_pereira2015_options_learning_nodes_bt.md) — *A Framework for Constrained and Adaptive Behavior-Based Agents*
- [08_iovino2021_gp_bt_unpredictable.md](./08_iovino2021_gp_bt_unpredictable.md) — *Learning Behavior Trees with Genetic Programming in Unpredictable Environments*
- [09_scheide2021_mcdags_grammar_bt.md](./09_scheide2021_mcdags_grammar_bt.md) — *Behavior Tree Learning for Robotic Task Planning through Monte Carlo DAG Search over a Formal Grammar*
- [10_anne2023_game_map_elites_bt.md](./10_anne2023_game_map_elites_bt.md) — *GAME: Generational Adversarial MAP-Elites for Co-evolving Behavior Trees*
- [11_ozkahraman2020_cbf_bt.md](./11_ozkahraman2020_cbf_bt.md) — *Combining Control Barrier Functions and Behavior Trees for Multi-Agent Underwater Coverage Missions*
- [12_serbinowska2025_nsbt_verification.md](./12_serbinowska2025_nsbt_verification.md) — *Neuro-Symbolic Behavior Trees (NSBTs) and Their Verification*
- [13_hemono2026_automatic_bt_generation_hrc.md](./13_hemono2026_automatic_bt_generation_hrc.md) — *Automatic Behavior Tree Generation for Enhanced Human–Robot Collaborative Task Planning in Industry 5.0: A Systematic Review*
- [14_bae2025_continuous_crowd_locomotion_crowdes.md](./14_bae2025_continuous_crowd_locomotion_crowdes.md) — *Continuous Locomotive Crowd Behavior Generation*
- [15_fu2025_moflow_onestep_flow_matching.md](./15_fu2025_moflow_onestep_flow_matching.md) — *MoFlow: One-Step Flow Matching for Human Trajectory Forecasting via Implicit Maximum Likelihood Estimation based Distillation (CVPR 2025)*
- [16_xing2025_goalflow_multimodal_trajectories.md](./16_xing2025_goalflow_multimodal_trajectories.md) — *GoalFlow: Goal-Driven Flow Matching for Multimodal Trajectories Generation in End-to-End Autonomous Driving (CVPR 2025)*
- [17_yan2025_trajflow_motion_prediction.md](./17_yan2025_trajflow_motion_prediction.md) — *TrajFlow: Multi-modal Motion Prediction via Flow Matching (IROS 2025)*
- [18_ye2024_tcfm_trajectory_conditional_flow_matching.md](./18_ye2024_tcfm_trajectory_conditional_flow_matching.md) — *Efficient Trajectory Forecasting and Generation with Conditional Flow Matching (IROS 2024)*
- [19_zhu2025_motion_field_regularized_flow_matching.md](./19_zhu2025_motion_field_regularized_flow_matching.md) — *Plausible and Feasible Long-Term Human Trajectory Prediction via Motion Field-Regularized Flow Matching (AAAI 2025)*
- [20_tan2025_flow_planner_interactive_behavior.md](./20_tan2025_flow_planner_interactive_behavior.md) — *Flow Matching-Based Autonomous Driving Planning with Advanced Interactive Behavior Modeling (NeurIPS 2025)*
- [21_mao2026_low_rank_spectral_flow_matching.md](./21_mao2026_low_rank_spectral_flow_matching.md) — *Low-Rank Spectral Flow Matching for Human Trajectory Prediction (KDD 2026)*

### Technical Reports & Deep Dives
- [design/README.md](./design/README.md) — *Mechanics-Based Architecture & Design Hub: Learnable Behavior Trees from Continuous Flow Models (Flow2BT)*
- [TREEFLOW_AND_FLOW_TO_BT_DISTILLATION.md](./TREEFLOW_AND_FLOW_TO_BT_DISTILLATION.md) — *Deep Dive: TreeFlow and the Distillation of Continuous Flows into Learnable Behavior Trees*
- [SURVEY_SYNTHESIS_AND_COMPENDIUM.md](./SURVEY_SYNTHESIS_AND_COMPENDIUM.md) — *Comprehensive Monolithic Compendium of All Survey Dossiers & Matrices*


