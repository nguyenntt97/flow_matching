# Subsystem 5: Action Leaf Parameterization via Dynamical Movement Primitives (DMPs)

**File Location:** `/home/nguyen/projects/flow_matching/survey/design/05_action_leaf_dmp_compression.md`  
**Subsystem Role:** Compressing high-dimensional, unimodal continuous flow branches into fast, certified, closed-form Dynamical Movement Primitives executing at $> 500\,\text{Hz}$.

---

> **Measured corrections (2026-10-01).** The design below is kept as originally proposed. Where the implementation in `src/flow2bt/` and `src/runtime/` contradicted it, the measured result is listed here; section numbers (§) refer to [IMPLEMENTATION_FINDINGS.md](./IMPLEMENTATION_FINDINGS.md).
>
> - **Integration step.** Semi-implicit Euler is unstable at the simulator's 0.2 s step (K=100, tau=2 s); the leaves must be sub-stepped (§4.1). The runtime ticks at **50 ms (20 Hz), not 10 ms / 100 Hz** (§9.3).
> - **Basis count vs. samples.** A 2 s leaf at 5 fps is 11 points. Fitting 20 bases to them made replay error *worse* (2.2 → 8.9 cm), so demonstrations are spline-resampled onto the integration grid first (§4.2).
> - **`g_ℓ = c_{t,nav}` (section 1.1 below) does not work**: the waypoint recedes. Each execution uses a fixed goal (§6.2).
> - **The lifecycle is load-bearing.** Without re-entry on SUCCESS / phase timeout, 71 of 406 agents finished and Travel Time EMD was 11.9 (§6.3).
> - **The forcing term must be evaluated in the fitted (navmesh) frame and rotated out.** Applying fitted-frame weights to world axes (with the `g − x0` clamp) put 0.5 m of error on replays; fixed in `DMPBank.acceleration` (§9.4).
> - **`||x − g|| ≤ ε → SUCCESS` (section 1.3 below) is degenerate for a stop leaf**: its goal lies within ε of its entry. Stationary leaves now end by time instead: RUNNING for tau, then FAILURE for a refractory tau, so the enclosing Fallback falls through to a walking subtree (§9.6).
> - **The mode table in section 1.2 below is not what is fitted.** Leaves are regressed from induced clusters, not hand-designed (no `D = 4√K` stop, no 2.5x tau yield).
> - **Measured replay error vs. each leaf's prototype: 0–7 cm** at the 50 ms tick. Executing a leaf as a DMP adds no measurable open-loop error over its prototype (routed ADE 0.350 vs 0.352) (§9.2).

---

## 1. Mathematical Mechanics & Functional Role

In complex continuous control, running a multi-million parameter neural network (e.g. UNet, Transformer, or numerical ODE solver) at every control tick incurs high computational latency ($> 20 - 100\,\text{ms}$) and consumes heavy GPU resources.

Subsystem 5 solves this by exploiting the theoretical guarantee of **TreeFlow** ([Ramachandran & Sra, 2026](../01_ramachandran2026_trees_to_flows.md)): once conditioned on a specific tree leaf partition $\ell$, the trajectory distribution $\Xi_\ell$ is **unimodal and low-curvature**. Each leaf can therefore be distilled into a **Dynamical Movement Primitive (DMP)**:

```
[ Unimodal Trajectory Cluster Xi_ell from Subsystem 3 ]
                           │
                           ▼  Centroid at tau = 1
         Attractor Goal Position: g_ell = c_{t, nav}
                           │
                           ▼  Linear Ridge Regression
            Non-Linear Forcing Function Weights: w_DMP
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│      DMP SECOND-ORDER CANONICAL DYNAMICAL SYSTEM       │
│                                                        │
│   tau * dv/dt = K * (g_ell - x) - D * v + f_ell(s)     │
│   tau * dx/dt = v                                      │
│                                                        │
│   Evaluated in < 0.1 ms on CPU (> 500 Hz control)      │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼  Status Lifecycle
           ||x - g_ell|| <= eps  ──►  SUCCESS
           Blocked / Timeout     ──►  FAILURE
           En route              ──►  RUNNING (emit control u)
```

### 1.1 DMP Mathematical Formulation
A discrete DMP consists of two coupled systems:
1. **Canonical System (Phase Clock):**
   $$\tau \dot{s} = -\alpha_s s, \quad s(0) = 1, \quad s \in [0, 1]$$
   where $s$ acts as a decay phase variable replacing explicit time, and $\alpha_s$ is a decay constant.
2. **Transformation System:**
   $$\begin{aligned}
   \tau \dot{\mathbf{v}} &= K (\mathbf{g}_\ell - \mathbf{x}) - D \mathbf{v} + (\mathbf{g}_\ell - \mathbf{x}_0) \mathbf{f}_\ell(s) \\
   \tau \dot{\mathbf{x}} &= \mathbf{v}
   \end{aligned}$$
   where $\mathbf{x}, \mathbf{v} \in \mathbb{R}^2$ are the pedestrian position and velocity, $\mathbf{g}_\ell \in \mathbb{R}^2$ is the goal attractor from Subsystem 1 ($\mathbf{g}_\ell = \mathbf{c}_{t, \text{nav}}$), $K$ is the spring stiffness, $D = 2\sqrt{K}$ is critical damping, and $\tau$ is the execution time scaling factor.
3. **Non-Linear Forcing Function:**
   $$\mathbf{f}_\ell(s) = \frac{\sum_{i=1}^P \psi_i(s) \mathbf{w}_i}{\sum_{i=1}^P \psi_i(s)} s, \quad \psi_i(s) = \exp\left(-h_i (s - c_i)^2\right)$$
   where $\psi_i(s)$ are Gaussian basis functions spaced along the phase trajectory. The shape weights $\mathbf{W}_\ell = [\mathbf{w}_1, \dots, \mathbf{w}_P] \in \mathbb{R}^{2 \times P}$ are fit to the demonstration trajectories $\Xi_\ell$ via closed-form linear ridge regression.

### 1.2 Mapping to the $B=8$ Behavioral Modes of Pedestrian Locomotion
Following the empirical discovery in [Bae et al. (2025)](../14_bae2025_continuous_crowd_locomotion_crowdes.md), pedestrian locomotion decomposes into 8 discrete action primitives:

| Behavioral Mode | DMP Goal Setting ($\mathbf{g}_\ell$) | Forcing Term $\mathbf{f}_\ell(s)$ | Execution Characteristics |
| :--- | :--- | :--- | :--- |
| **1. Straight March** | $\mathbf{c}_{t, \text{nav}}$ (on NavMesh polyline) | $\approx 0$ (pure linear spring-damper) | Maximum pace $\nu$, straight tracking |
| **2. Swerve Left** | $\mathbf{c}_{t, \text{nav}} + d_{\text{lat}} \cdot \mathbf{n}_{\text{left}}$ | Smooth lateral curve | Evades obstacle on right side |
| **3. Swerve Right** | $\mathbf{c}_{t, \text{nav}} + d_{\text{lat}} \cdot \mathbf{n}_{\text{right}}$ | Smooth lateral curve | Evades obstacle on left side |
| **4. Yield / Slow Down** | $\mathbf{c}_{t, \text{nav}}$ | $\tau$ scaled by $2.5\times$ | Reduces pace to allow oncoming walker to pass |
| **5. Stop & Wait** | $\mathbf{c}_t$ (current position) | Damping increased ($D = 4\sqrt{K}$) | Velocity actively arrested to zero |
| **6. Accelerate Overtake**| $\mathbf{c}_{t, \text{nav}} + \Delta \mathbf{c}_{\text{forward}}$ | Forward velocity boost | Overtakes slower pedestrian ahead |
| **7. Group Cohesion Walk**| Weighted centroid of group neighbors | Attraction toward neighbor positions | Maintains flocking / group formation |
| **8. Backstep Recovery** | $\mathbf{c}_t - \Delta \mathbf{c}_{\text{forward}}$ | Reverse velocity profile | Backs away from deadlock or door collision |

### 1.3 Behavior Tree Action Lifecycle Implementation
Each DMP leaf exposes a standardized `tick()` method conforming to BT semantics:
```python
def tick(self, state: PedestrianState) -> NodeStatus:
    # 1. Update phase clock
    self.s += (-self.alpha_s * self.s / self.tau) * dt

    # 2. Integrate DMP equations
    f = self.evaluate_forcing(self.s)
    a = (self.K * (self.g - state.pos) - self.D * state.vel + f) / self.tau
    state.vel += a * dt
    state.pos += state.vel * dt

    # 3. Check termination status
    if np.linalg.norm(state.pos - self.g) <= self.epsilon:
        return NodeStatus.SUCCESS
    elif state.is_collision or state.elapsed_time > self.timeout:
        return NodeStatus.FAILURE
    else:
        return NodeStatus.RUNNING
```

---

## 2. Design Choices & Comparisons from Previous Work

| Approach | Action Primitive Representation | Latency | Guarantees | Why Chosen / Adapted in Flow2BT |
| :--- | :--- | :--- | :--- | :--- |
| **Recurrent Trajectory Predictor $\mu_\varphi$** ([Bae et al., 2025](../14_bae2025_continuous_crowd_locomotion_crowdes.md)) | Neural network with cross-attention | $\approx 20 - 40\,\text{ms}$ per agent | No stability proofs; outputs can diverge | Replaced by DMPs: neural prediction is too slow for 100+ agents |
| **SMDP Options with Q-Learning** ([Pereira & Engel, 2015](../07_pereira2015_options_learning_nodes_bt.md)) | Discrete/continuous action Q-learners | Low | Empirical convergence; lacks smooth trajectory shapes | Concept of options as subtrees is retained, but parameterized by DMPs |
| **Dynamical Movement Primitives in BTs** ([Chatzilygeroudis et al., 2021](../06_chatzilygeroudis2021_bt_movement_skills.md)) | Spring-damper ODE with learned forcing terms | **$< 0.1\,\text{ms}$** | **Provably stable**: asymptotically converges to $\mathbf{g}_\ell$ | **Adopted as Subsystem 5**: Provides guaranteed convergence and sub-millisecond execution |
| **Linear Feedback / LQR Leaves** ([Sprague & Ögren, 2022](../05_sprague2022_neural_controllers_bt.md)) | Gain matrix $\mathbf{u} = -\mathbf{K} (\mathbf{x} - \mathbf{x}^\star)$ | Ultra-fast | Requires known linear dynamics | Complementary: DMPs generalize LQR by allowing non-linear forcing shapes |

---

## 3. Interface with Downstream Subsystems

1. **To Subsystem 6 (Reactive BT Assembly):** DMPs instantiate the terminal Action leaves ($A \in V_T$) that return `RUNNING`, `SUCCESS`, or `FAILURE`.
2. **To Subsystem 7 (Safety & Verification):** The DMP commanded acceleration $\mathbf{a}_{\text{DMP}}$ is fed directly into the Control Barrier Function Quadratic Program (CBF-QP) as the reference nominal control input:
   $$\mathbf{u}^\star = \arg\min_{\mathbf{u}} \|\mathbf{u} - \mathbf{u}_{\text{DMP}}\|^2 \quad \text{s.t.} \quad \dot{h}(\mathbf{s}, \mathbf{u}) + \alpha(h(\mathbf{s})) \ge 0$$

---

## 4. Codebase Implementation & Correspondences

| Architectural Component | Implementation File | Key Class / Function | Operational Mechanics & Settings |
| :--- | :--- | :--- | :--- |
| **DMP Attractor System** | [`src/flow2bt/primitives.py`](../../src/flow2bt/primitives.py) | `DynamicMovementPrimitive` | Critically damped 2nd-order ODE ($K=100$, $D=20$) with exponential canonical phase decay $\tau \dot{s} = -\alpha_s s$ ($\alpha_s = 4.0$). |
| **Forcing Term Fitting** | [`src/flow2bt/primitives.py`](../../src/flow2bt/primitives.py) | `fit_dmp_from_cluster` | Fits nonlinear forcing function $\mathbf{f}(s)$ using $N_{\text{basis}}=10$ Gaussian radial basis kernels via linear ridge regression over each unimodal leaf cluster. |
| **Goal Anchoring Mechanism** | [`src/flow2bt/primitives.py`](../../src/flow2bt/primitives.py), [`src/runtime/controllers.py`](../../src/runtime/controllers.py) | Goal initialization in `step()` | Sets fixed goal $\mathbf{g}_\ell = \mathbf{p}_i(t_0) + \mathbf{R}_i \Delta\mathbf{p}_\ell^{\text{cluster}}$. Crucial fix: prevents spring acceleration blowup ($> 45\,\text{m/s}^2$) caused by receding waypoints. |
| **High-Frequency Numerical Integration** | [`src/flow2bt/primitives.py`](../../src/flow2bt/primitives.py) | Semi-implicit Euler integration | Integrates at $\Delta t = 10\,\text{ms}$ ($100\,\text{Hz}$), satisfying $\Delta t < 2/\sqrt{K} = 0.2\,\text{s}$ and providing sub-$1.7\,\text{cm}$ tracking error. |
| **BT Leaf Action Wrapper** | [`src/flow2bt/bt.py`](../../src/flow2bt/bt.py), [`src/flow2bt/primitives.py`](../../src/flow2bt/primitives.py) | `DMPActionNode` | Manages node lifecycle: returns `RUNNING` while phase $s > 0.05$ and distance $> \epsilon$, and `SUCCESS` upon goal arrival, prompting root re-evaluation. |


