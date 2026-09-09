# Learning of Parameters in Behavior Trees for Movement Skills

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
- **Interpretability Preservation:** Fully preserves native BT interpretability: the tree structure remains completely symbolic and human-readable. Engineers can inspect, tune, or override individual learned parameter bounds (e.g. maximum allowable force threshold $\theta_{\text{force}} = 12\,\text{N}$).
- **Scalability & Gradient Stability:** Black-box policy search (Bayesian Optimization / CMA-ES) avoids non-differentiable gradient instability but suffers from the curse of dimensionality, restricting the total number of tunable parameters to $D \le 30$. High-dimensional neural representations cannot be trained end-to-end using this formulation.
- **Unaddressed Gaps:** Topology learning: the tree structure must be provided a priori by an expert designer; the algorithm cannot discover new control branches or structurally synthesize fallback recovery loops from data.

