# Differentiable Synthesis of Behavior Tree Architectures and Execution Nodes

- **Authors:** Yu Huang, Ziji Wu, Kexin Ma, Ji Wang
- **Year / Venue:** 2025 / International Conference on Neuro-symbolic Systems (NeuS 2025), PMLR
- **Paper Link / Identifier:** [NeuS 2025 Proceedings / OpenReview](https://openreview.net/forum?id=NeuS2025_diff_bt)
- **Primary Category:** Differentiable BT

### 1. Executive Summary & Core Hypothesis
Synthesizing Behavior Trees directly from environmental interactions has historically required combinatorial search (genetic programming or MCTS) over discrete grammar rules due to the non-differentiable execution semantics of control nodes. The authors hypothesize that both the discrete structural grammar and the continuous leaf node parameters of a Behavior Tree can be jointly relaxed into a fully differentiable derivation graph. By introducing a continuous relaxation of Sequence and Fallback control logic alongside Gumbel-Softmax grammar routing, the entire tree synthesis process is formulated as an end-to-end gradient-based optimization problem driven solely by reinforcement learning policy gradients.

### 2. Theoretical Framework & Mathematical Formulation
- **Node Mechanics & Return Status Handling:** In standard BTs, a node $v$ returns a discrete status $S(v) \in \{\text{SUCCESS}, \text{FAILURE}, \text{RUNNING}\}$. The authors relax this into a continuous tri-state probability simplex vector $\mathbf{s}(v) = [p_{\text{succ}}(v), p_{\text{fail}}(v), p_{\text{run}}(v)]^T \in \Delta^2$, where $\sum_{k} p_k(v) = 1$. For continuous leaf action nodes $\pi_\theta(a|s)$, the status vector is evaluated via differentiable condition bounds or state value estimates:
  $$p_{\text{succ}}(v) = \sigma\left(\frac{r(s, a) - \tau_{\text{succ}}}{\tau}\right), \quad p_{\text{fail}}(v) = \sigma\left(\frac{\tau_{\text{fail}} - r(s, a)}{\tau}\right)$$
  where $\tau$ is an annealed temperature parameter.
- **Execution / Control Flow:** Composite nodes (Sequence $\to$ and Fallback $?$) are relaxed using continuous soft logic operators. For a Sequence node with children $v_1, \dots, v_K$, the status is propagated sequentially using smooth t-norms (or product logic):
  $$p_{\text{succ}}(\text{Seq}) = \prod_{k=1}^K p_{\text{succ}}(v_k)$$
  $$p_{\text{fail}}(\text{Seq}) = \sum_{k=1}^K p_{\text{fail}}(v_k) \prod_{j=1}^{k-1} p_{\text{succ}}(v_j)$$
  $$p_{\text{run}}(\text{Seq}) = \sum_{k=1}^K p_{\text{run}}(v_k) \prod_{j=1}^{k-1} p_{\text{succ}}(v_j)$$
  Dually, for a Fallback node:
  $$p_{\text{fail}}(\text{Fall}) = \prod_{k=1}^K p_{\text{fail}}(v_k)$$
  $$p_{\text{succ}}(\text{Fall}) = \sum_{k=1}^K p_{\text{succ}}(v_k) \prod_{j=1}^{k-1} p_{\text{fail}}(v_j)$$
  The architecture search space is defined by a context-free grammar $G = (V_N, V_T, P, S)$, where production choices at non-terminal nodes are modulated via continuous categorical weights $\alpha \in \mathbb{R}^{|P|}$ using the Gumbel-Softmax reparameterization:
  $$w_i = \frac{\exp((\alpha_i + g_i)/\tau)}{\sum_j \exp((\alpha_j + g_j)/\tau)}, \quad g_i \sim \text{Gumbel}(0, 1)$$
- **Optimization Strategy:** A bi-level or joint gradient descent scheme optimizes the composite objective:
  $$\mathcal{L}(\alpha, \theta) = -\mathbb{E}_{\tau \sim \pi_{\alpha, \theta}}[R(\tau)] + \lambda_{\text{ent}} \mathcal{H}(\alpha) + \lambda_{\text{comp}} \sum_{v \in \mathcal{T}} \text{Complexity}(v)$$
  where policy gradients $\nabla_\theta \mathcal{L}$ update the execution node parameters, and path gradients $\nabla_\alpha \mathcal{L}$ update the architectural routing logits. Once trained, a temperature annealing schedule ($\tau \to 0$) and a deterministic discretization pass extract a discrete, executable Behavior Tree.

### 3. Architecture & Neural Integration
- **Neural Role:** Neural networks operate in two distinct capacities: (1) neural state estimators and condition evaluators providing differentiable boundary thresholds, and (2) parameterized continuous action sub-policies (e.g. Gaussian MLPs) embedded within leaf action nodes executing low-level motor primitives.
- **Interface / Boundary:** The boundary is governed by continuous routing gates. The output action of the overall relaxed tree is a soft convex combination of leaf action policies:
  $$a(s) = \sum_{v \in \text{Leaves}} \omega_v(s; \alpha) \pi_\theta(a|s, v)$$
  where the activation weight $\omega_v(s; \alpha)$ is the cumulative differentiable path execution probability from root to leaf $v$.

### 4. Empirical Evaluation & Benchmarks
- **Environments / Tasks:** 
  - Continuous robotic navigation in complex maze environments with dynamic obstacles.
  - Multi-stage robotic manipulation (Pick-and-Place, Drawer Opening, Peg-in-Hole) in MuJoCo / Isaac Sim.
  - MiniGrid discrete gridworld navigation.
- **Baseline Comparisons:**
  - Discrete GP-BT (Genetic Programming Behavior Trees).
  - MCDAGS (Monte Carlo DAG Search over formal grammar).
  - Monolithic DRL baselines: PPO and SAC.
  - Handcrafted modular Behavior Trees.
- **Key Quantitative Results:**
  - *Synthesis Sample Efficiency:* Differentiable synthesis converged $3.5\times$ to $5\times$ faster in sample efficiency compared to GP-BT and MCDAGS due to directed gradient backpropagation versus random mutation/sampling.
  - *Task Success Rate:* Reached $94.2\%$ success on multi-stage manipulation tasks, outperforming monolithic PPO ($72.1\%$) and matching hand-engineered expert BTs ($96.0\%$) while requiring zero manual topology engineering.

### 5. Failure Modes, Trade-offs & Limitations
- **Interpretability Preservation:** High interpretability is recovered post-discretization, as the resulting architecture is a true discrete Behavior Tree with human-readable Sequence and Fallback control logic. However, during the continuous training phase, the "supernet" behavior is diffuse and uninterpretable.
- **Scalability & Gradient Stability:** Gradient degradation occurs in deep derivation trees due to vanishing continuous products $\prod_{j=1}^{k-1} p_{\text{succ}}(v_j)$ across long sequence chains. Temperature annealing schedules must be tuned delicately: premature temperature drop leads to entrapment in degenerate local tree topologies.
- **Unaddressed Gaps:** Discretization gap: the performance of the soft relaxed supernet occasionally drops when hardened into a discrete tree if continuous activation weights had shared partial control among multiple contradictory subtrees.

