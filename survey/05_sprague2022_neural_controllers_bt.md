# Adding Neural Network Controllers to Behavior Trees without Destroying Performance Guarantees

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

