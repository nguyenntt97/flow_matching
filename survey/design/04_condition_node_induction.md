# Subsystem 4: Condition Node Induction & State-Space Partitioning

**File Location:** `/home/nguyen/projects/flow_matching/survey/design/04_condition_node_induction.md`  
**Subsystem Role:** Training fast, interpretable condition predicates at each topological bifurcation node to dynamically route Behavior Tree execution at runtime.

---

## 1. Mathematical Mechanics & Functional Role

Subsystem 3 reveals *where* trajectories diverge into different strategies, but does not provide an online decision rule. When an autonomous agent or simulated pedestrian moves through an environment, it cannot integrate future trajectories to decide which branch to take.

Subsystem 4 constructs explicit, real-time **Condition Nodes** $C_k(\mathbf{s}) \in \{\text{SUCCESS}, \text{FAILURE}\}$ by fitting separating hyperplanes in the agent's observation state space:

```
At Bifurcation Node k from Subsystem 3:
Left Cluster Initial States: S_left = {s_0^(i) | xi_i in Xi_left}
Right Cluster Initial States: S_right = {s_0^(j) | xi_j in Xi_right}
                              │
                              ▼  Support Vector Machine / Logistic Fit
Optimal Separating Hyperplane: w_k^T phi(s) + b_k = 0
                              │
                              ├──────────────────────────────────┐
                              ▼                                  ▼
                [ Hard Evaluation (Deployment) ]    [ Soft Evaluation (Fine-Tuning) ]
                C_k(s) = I(w_k^T phi(s) + b_k >= 0)  p_k(s) = sigma(beta(w^T phi(s) + b))
                Ticked in < 0.001 ms (> 500 Hz)     Differentiable backprop (Huang 2025)
```

### 1.1 State Representation ($\mathbf{s}$)
The condition node is evaluated on feature vector $\phi(\mathbf{s}) \in \mathbb{R}^d$ capturing immediate local geometry and social context:
$$\phi(\mathbf{s}) = \begin{bmatrix}
\mathbf{p}_{\text{obs}} - \mathbf{c}_t & \text{(Relative vector to closest obstacle)} \\
\mathbf{v}_{\text{rel}} = \mathbf{v}_{\text{neighbor}} - \mathbf{v}_t & \text{(Relative velocity of oncoming pedestrian)} \\
\text{TTC} = \frac{\|\mathbf{p}_{\text{neighbor}} - \mathbf{c}_t\|}{\|\mathbf{v}_{\text{rel}}\|} & \text{(Time-to-Collision estimate)} \\
\theta_{\text{nav}} = \angle(\mathbf{c}_{t, \text{nav}} - \mathbf{c}_t) & \text{(Heading error to NavMesh polyline)} \\
d_{\text{corridor}} & \text{(Lateral clearance to left/right boundaries)}
\end{bmatrix}$$

### 1.2 Optimization Objective: Maximum-Margin Separation
At each bifurcation node $k$, we solve a soft-margin Support Vector Machine (SVM) formulation:
$$\min_{\mathbf{w}_k, b_k, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}_k\|_2^2 + C \sum_{i=1}^{M_k} \xi_i \quad \text{s.t.} \quad y_i (\mathbf{w}_k^\top \phi(\mathbf{s}_i) + b_k) \ge 1 - \xi_i, \quad \xi_i \ge 0$$
where $y_i = +1$ for trajectories taking the left branch and $y_i = -1$ for trajectories taking the right branch.

### 1.3 Operational Modes
1. **Deterministic BT Tick (Hardware / Production):**
   $$C_k(\mathbf{s}) = \begin{cases} \text{SUCCESS} & \text{if } \mathbf{w}_k^\top \phi(\mathbf{s}) + b_k \ge 0 \\ \text{FAILURE} & \text{otherwise} \end{cases}$$
   *Computational complexity:* A single dot product of dimension $d \approx 10$. Evaluation takes less than $1\,\mu\text{s}$, allowing BT ticks at $> 1000\,\text{Hz}$.
2. **Soft Differentiable Routing (Training / Fine-Tuning):**
   Following [Frosst & Hinton (2017)](file:///home/nguyen/projects/flow_matching/survey/04_frosst2017_soft_decision_trees.md) and [Huang et al. (2025)](file:///home/nguyen/projects/flow_matching/survey/02_huang2025_differentiable_bt_synthesis.md):
   $$p_k(\mathbf{s}) = \sigma\left(\beta (\mathbf{w}_k^\top \phi(\mathbf{s}) + b_k)\right) \in [0, 1]$$
   where $\beta$ is an inverse temperature parameter annealed toward infinity ($\beta \to \infty$) during hardening.

---

## 2. Design Choices & Comparisons from Previous Work

| Approach | Routing Mechanism | Computational Cost | Failure Handling | Why Chosen / Adapted in Flow2BT |
| :--- | :--- | :--- | :--- | :--- |
| **Markov Transition Matrix** ([Bae et al., 2025](file:///home/nguyen/projects/flow_matching/survey/14_bae2025_continuous_crowd_locomotion_crowdes.md)) | Neural network $\mu_\phi$ sampling $b_f \sim P(b_f \mid \cdot)$ every $4\,\text{s}$ | Moderate ($\approx 10\,\text{ms}$) | **Stochastic**: may fail to trigger evasion when an obstacle appears | Rejected for online control: 4-second delay causes crashes |
| **Deep Neural Condition Leaves** ([Serbinowska et al., 2025](file:///home/nguyen/projects/flow_matching/survey/12_serbinowska2025_nsbt_verification.md)) | Deep MLP evaluating raw camera/occupancy grids | High ($\approx 15 - 30\,\text{ms}$) | Requires polyhedral over-approximations for verification | Supported for high-dimensional vision, but linear hyperplanes preferred for locomotion |
| **Soft Decision Tree Gating** ([Frosst & Hinton, 2017](file:///home/nguyen/projects/flow_matching/survey/04_frosst2017_soft_decision_trees.md)) | Logistic sigmoid gates $\sigma(\mathbf{w}^\top \mathbf{x} + b)$ | Ultra-fast ($< 1\,\mu\text{s}$) | Soft co-activation during training | **Adapted as Subsystem 4**: Linear hyperplanes provide mathematical interpretability and instant evaluation |
| **Gumbel-Softmax Grammar Logits** ([Huang et al., 2025](file:///home/nguyen/projects/flow_matching/survey/02_huang2025_differentiable_bt_synthesis.md)) | Categorical distribution over grammar choices | Low | Suffers from discretization gap upon hardening | **Adapted for Fine-Tuning**: Used only for local boundary calibration |

---

## 3. Interface with Downstream Subsystems

1. **To Subsystem 6 (Reactive BT Assembly):** Instantiates the terminal Condition leaves ($C \in V_T$) placed as guards inside Sequence and Fallback subtrees.
2. **To Subsystem 7 (Formal Verification):** The linear hyperplane $\mathbf{w}_k^\top \phi(\mathbf{s}) + b_k = 0$ translates into linear constraints easily ingested by SMT solvers (Z3 / MathSAT) and the BehaVerify model checker.

