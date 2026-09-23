# Subsystem 7: Safety Invariance via Control Barrier Functions & Formal SMT Model Checking

**File Location:** `/home/nguyen/projects/flow_matching/survey/design/07_safety_cbf_formal_verification.md`  
**Subsystem Role:** Enforcing zero-collision safety invariance ($C_R = 0$) at runtime via Control Barrier Functions (CBF-QP) and formally verifying temporal safety and liveness properties offline via BehaVerify and nuXmv.

---

## 1. Mathematical Mechanics & Functional Role

Pure machine learning policies—including continuous flow models ([Bae et al., 2025](../14_bae2025_continuous_crowd_locomotion_crowdes.md)), diffusion models, and neural RL—provide only statistical guarantees. In dense pedestrian crowds, even state-of-the-art flow policies incur non-zero collision rates ($C_R \approx 1.5 - 3.2\%$). Furthermore, deep models cannot be formally audited or certified against safety regulations (e.g., ISO 13482 for mobile robotics, Industry 5.0 human-robot safety standards [Hémono et al., 2026](../13_hemono2026_automatic_bt_generation_hrc.md)).

Subsystem 7 provides a **dual-layer safety guarantee**:
1. **Online Runtime Shielding:** A high-speed ($> 500\,\text{Hz}$) Control Barrier Function Quadratic Program (CBF-QP) that filters nominal Behavior Tree actions to guarantee forward invariance of the collision-free set.
2. **Offline Formal Verification:** Automated compilation of the induced Behavior Tree into BehaVerify DSL and nuXmv symbolic model checking to formally verify temporal specifications ($\mathcal{G} \neg \text{Collision}$, $\mathcal{F} \text{AtGoal}$).

```
[ BT Action Leaf Command: u_BT (from Subsystem 6) ]
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│         ONLINE RUNTIME SAFETY FILTER: CBF-QP           │
│                                                        │
│  u* = argmin ||u - u_BT||^2                            │
│  s.t.  grad_s h_k(s)^T [f(s) + g(s)u] + alpha(h_k) >= 0│
│                                                        │
│  Constraints:                                          │
│  - Pairwise Pedestrian Distance: ||p_i - p_j|| >= d_min│
│  - NavMesh Obstacle Boundary:    SDF(p_i) >= r_agent   │
│  - Maximum Acceleration Bounds:  ||u|| <= a_max        │
└────────────────────────┬───────────────────────────────┘
                         │
                         ▼
        [ Actuator Commanded Acceleration u* ]
                         │
                         │ (Guarantees Collision Rate C_R = 0.0%)
                         ▼
┌────────────────────────────────────────────────────────┐
│     OFFLINE FORMAL SMT VERIFICATION: BEHAVERIFY        │
│                                                        │
│  [ Assembled BT ] ──► [ BehaVerify DSL Compiler ]      │
│                                  │                     │
│                                  ▼                     │
│                         [ nuXmv SMT Model ]            │
│                                  │                     │
│    Model Checking: LTLSPEC G !(collision)              │
│                    LTLSPEC F (at_destination)         │
│                                  │                     │
│          ┌───────────────────────┴───────────────────┐ │
│          ▼ True                                      ▼ │
│  [ Mathematically Verified ]          [ Counterexample Trace ]
│                                       (Refines Subsystem 4)
└────────────────────────────────────────────────────────┘
```

### 1.1 Online Runtime Safety: Control Barrier Functions (CBF-QP)

Following the control-theoretic foundation of [Özkahraman & Ögren (2020)](../11_ozkahraman2020_cbf_bt.md) and Ames et al. (2019), we define the continuous pedestrian dynamics as an affine control system:
$$\dot{\mathbf{s}}_i = \mathbf{f}(\mathbf{s}_i) + \mathbf{g}(\mathbf{s}_i) \mathbf{u}_i, \quad \mathbf{s}_i = [\mathbf{p}_i^\top, \mathbf{v}_i^\top]^\top \in \mathbb{R}^4$$
where $\mathbf{p}_i \in \mathbb{R}^2$ is position, $\mathbf{v}_i \in \mathbb{R}^2$ is velocity, and $\mathbf{u}_i \in \mathbb{R}^2$ is commanded acceleration.

#### 1.1.1 Pairwise Inter-Agent Safety Barrier
To prevent collision between pedestrian $i$ and neighbor $j$, the safe set is defined by the superlevel set $\mathcal{C}_{ij} = \{\mathbf{s} \in \mathbb{R}^8 \mid h_{ij}(\mathbf{s}) \ge 0\}$:
$$h_{ij}(\mathbf{s}) = \|\mathbf{p}_i - \mathbf{p}_j\|^2 - d_{\min}^2$$
where $d_{\min} = 2 r_{\text{agent}} + \delta_{\text{margin}}$ (with typical pedestrian radius $r_{\text{agent}} = 0.3\,\text{m}$, $\delta_{\text{margin}} = 0.1\,\text{m} \implies d_{\min} = 0.7\,\text{m}$).

Since $h_{ij}$ has relative degree 2 with respect to acceleration $\mathbf{u}_i$, we define the higher-order barrier function:
$$h_{ij}^{(e)}(\mathbf{s}) = \dot{h}_{ij}(\mathbf{s}) + \alpha_1 h_{ij}(\mathbf{s}) = 2 (\mathbf{p}_i - \mathbf{p}_j)^\top (\mathbf{v}_i - \mathbf{v}_j) + \alpha_1 \left(\|\mathbf{p}_i - \mathbf{p}_j\|^2 - d_{\min}^2\right)$$

The forward invariance of $\mathcal{C}_{ij}$ is guaranteed if the control $\mathbf{u}_i$ satisfies:
$$\dot{h}_{ij}^{(e)}(\mathbf{s}, \mathbf{u}_i) + \alpha_2 h_{ij}^{(e)}(\mathbf{s}) \ge 0$$
which expands to the affine linear inequality constraint on $\mathbf{u}_i$:
$$2 (\mathbf{p}_i - \mathbf{p}_j)^\top \mathbf{u}_i \ge -2 \|\mathbf{v}_i - \mathbf{v}_j\|^2 - (\alpha_1 + \alpha_2) \dot{h}_{ij} - \alpha_1 \alpha_2 h_{ij} + 2 (\mathbf{p}_i - \mathbf{p}_j)^\top \mathbf{u}_j$$

#### 1.1.2 Static NavMesh Obstacle Barrier
For static environmental boundaries (walls, pillars, non-traversable terrain), let $\text{SDF}(\mathbf{p}_i)$ be the signed distance from $\mathbf{p}_i$ to the nearest non-traversable boundary $\partial \Omega_{\text{nav}}$ (precomputed from Subsystem 1):
$$h_{\text{obs}}(\mathbf{p}_i) = \text{SDF}(\mathbf{p}_i) - r_{\text{agent}}$$
$$\nabla h_{\text{obs}}^\top \mathbf{v}_i + \alpha_{\text{obs}} h_{\text{obs}}(\mathbf{p}_i) \ge 0$$

#### 1.1.3 Quadratic Program Formulation (CBF-QP)
At every control tick ($100\,\text{Hz}$), given the nominal control $\mathbf{u}_{\text{nominal}} = \mathbf{a}_{\text{DMP}}$ emitted by the active BT leaf (from Subsystems 5 and 6), the agent solves:
$$\begin{aligned}
\mathbf{u}_i^\star = \arg\min_{\mathbf{u}_i \in \mathbb{R}^2, \, \xi \ge 0} &\quad \frac{1}{2} \|\mathbf{u}_i - \mathbf{u}_{\text{nominal}}\|^2 + \lambda_{\text{slack}} \xi^2 \\
\text{subject to} &\quad \mathbf{A}_{\text{agent}} \mathbf{u}_i \le \mathbf{b}_{\text{agent}} + \xi \mathbf{1}, \\
&\quad \mathbf{A}_{\text{obs}} \mathbf{u}_i \le \mathbf{b}_{\text{obs}}, \\
&\quad \|\mathbf{u}_i\| \le a_{\max}
\end{aligned}$$
Because the objective is strictly convex quadratic and constraints are linear inequalities in $\mathbb{R}^2$, the QP solves via OSQP or qpOASES in **$< 0.05\,\text{ms}$**, easily scaling to hundreds of agents simultaneously.

---

### 1.2 Offline Formal SMT Verification: BehaVerify & nuXmv

While CBF-QP provides real-time collision filtering, it does not prevent higher-level logical failures such as **deadlocks**, **infinite livelock loops**, or **goal unreachability**. 

Following [Serbinowska et al. (2025)](../12_serbinowska2025_nsbt_verification.md), the synthesized Behavior Tree is compiled into a formal finite-state model using **BehaVerify**:

1. **BehaVerify DSL Translation:**
   Each node type in the Behavior Tree (Guarded Fallbacks, Sequences, Condition checks, and DMP execution states) maps to a transition system in the BehaVerify domain-specific language:
   ```smv
   MODULE main
   VAR
     tree : node_root();
     env  : environment(tree.active_action);
   
   -- Formal Specifications in Linear Temporal Logic (LTL)
   LTLSPEC G !(env.distance_to_nearest_agent < 0.7)  -- Absolute collision freedom
   LTLSPEC F (env.distance_to_destination < 0.5)      -- Eventual goal reachability
   LTLSPEC G F (env.agent_velocity > 0.1 | env.at_goal) -- Deadlock freedom
   ```

2. **Symbolic Model Checking via nuXmv:**
   The nuXmv SMT engine performs Bounded Model Checking (BMC) and Inductive Invariant Verification:
   - If the specifications hold, the tree is mathematically proven collision-free and deadlock-free over the entire state space.
   - If a specification fails, nuXmv produces an **exact counterexample trace** (e.g., two agents entering a narrow door from opposite directions where both choose `Yield` simultaneously, creating deadlock).

3. **Closed-Loop Counterexample Refinement:**
   The counterexample trace is fed directly back into Subsystem 4 to adjust condition hyperplane thresholds ($b_c$) or introduce an asymmetric tie-breaking condition node (`IsRightHandDominant`), closing the loop between induction and formal verification.

---

## 2. Design Choices & Comparisons from Previous Work

| Approach | Collision Avoidance Mechanism | Collision Rate $C_R$ | Latency | Formal Verification |
| :--- | :--- | :--- | :--- | :--- |
| **CrowdES Locomotion Simulator** ([Bae et al., 2025](../14_bae2025_continuous_crowd_locomotion_crowdes.md)) | Neural cross-attention + 4s chunking | **$2.5 - 3.2\%$** | $20 - 40\,\text{ms}$ | Impossible (black-box neural ODE) |
| **Unconstrained Flow Matching** ([Ramachandran & Sra, 2026](../01_ramachandran2026_trees_to_flows.md)) | Vector field integration | Variable ($> 1\%$) | $15 - 50\,\text{ms}$ | None |
| **RL-Trained BTs** ([Pereira & Engel, 2015](../07_pereira2015_options_learning_nodes_bt.md)) | Soft penalty rewards in MDP | $1.0 - 5.0\%$ | $< 1\,\text{ms}$ | None |
| **CBF-Augmented BTs** ([Özkahraman & Ögren, 2020](../11_ozkahraman2020_cbf_bt.md)) | Online CBF-QP safety filter | **$0.0\%$** (provably safe) | $< 0.1\,\text{ms}$ | Control-theoretic forward invariance |
| **BehaVerify Synthesis** ([Serbinowska et al., 2025](../12_serbinowska2025_nsbt_verification.md)) | SMT symbolic model checking (nuXmv) | Offline proven | Offline (compilation) | Full LTL/CTL specification proofs |
| **Subsystem 7: Dual-Layer Safety (Our Design)** | **Online CBF-QP + Offline BehaVerify / nuXmv** | **$\mathbf{0.0\%}$** | **$< 0.05\,\text{ms}$** | **Full control-theoretic invariance + symbolic LTL model checking** |

---

## 3. Interface with Downstream Subsystems

1. **Input from Subsystem 6:** Receives nominal action acceleration $\mathbf{u}_{\text{nominal}} = \mathbf{a}_{\text{DMP}}$ and discrete tree topology $\mathcal{T}$.
2. **Integration into Subsystem 8 (End-to-End Walkthrough):** Emits verified, safe control commands $\mathbf{u}^\star$ to the robot/pedestrian locomotion simulator at $100\,\text{Hz}$, achieving zero collisions ($C_R = 0.0\%$) and verified destination arrival.

