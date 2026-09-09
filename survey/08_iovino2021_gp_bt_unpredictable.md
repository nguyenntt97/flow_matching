# Learning Behavior Trees with Genetic Programming in Unpredictable Environments

- **Authors:** Matteo Iovino, Jonathan Styrud, Pietro Falco, Christian Smith
- **Year / Venue:** 2021 / IEEE International Conference on Robotics and Automation (ICRA 2021)
- **Paper Link / Identifier:** [IEEE Xplore / DOI: 10.1109/ICRA48506.2021.9561214](https://doi.org/10.1109/ICRA48506.2021.9561214) / [arXiv:2011.08528](https://arxiv.org/abs/2011.08528)
- **Primary Category:** Structure Evolution + Policy Gradient

### 1. Executive Summary & Core Hypothesis
Autonomous robots deployed in unstructured and stochastic environments must rapidly adapt their sequential decision logic to unmodeled external disturbances (such as human interference or mechanical slippage). The authors hypothesize that Genetic Programming (GP) can evolve robust, reactive Behavior Tree topologies from primitive modular building blocks directly within unpredictable environments. By incorporating stochastic disturbances during fitness evaluations and penalizing tree structural complexity (bloat), the evolutionary algorithm discovers compact, fault-tolerant control trees that significantly outperform both open-loop action plans and manually designed reactive trees.

### 2. Theoretical Framework & Mathematical Formulation
- **Node Mechanics & Return Status Handling:** Standard discrete Behavior Tree semantics $S \in \{\text{SUCCESS}, \text{FAILURE}, \text{RUNNING}\}$. Leaf nodes comprise atomic action primitives $A = \{a_1, \dots, a_m\}$ and condition primitives $C = \{c_1, \dots, c_n\}$. Action nodes execute continuous motor controllers for robot manipulation, returning RUNNING during execution and SUCCESS/FAILURE upon completion.
- **Execution / Control Flow:** Trees are constructed recursively from a terminal set $\mathcal{T} = A \cup C$ and a non-terminal composite function set $\mathcal{F} = \{\text{Sequence } (\to), \text{Fallback } (?)\}$. Control flow is strictly reactive: at each tick $\Delta t$, execution traverses from the root down the left-to-right hierarchy. If an action in progress is invalidated by an external environmental perturbation (e.g. an object knocked out of the gripper), upstream condition checks immediately fail on the subsequent tick, triggering fallback branches to re-grasp or re-align the object.
- **Optimization Strategy:** Evolutionary search over tree syntax graphs. A population of individuals $\mathcal{P} = \{BT_1, \dots, BT_N\}$ is initialized via ramped half-and-half tree generation. The multi-objective fitness function evaluates task performance while penalizing structural bloat:
  $$f(BT) = \frac{1}{K} \sum_{k=1}^K R_k(BT) - \alpha \cdot \text{Size}(BT) - \beta \cdot \text{Ticks}(BT)$$
  where $R_k(BT) \in [0, 1]$ is the task completion score in simulation trial $k$ subject to randomized stochastic disturbances, $\text{Size}(BT)$ is the total node count, $\text{Ticks}(BT)$ measures execution duration, and $\alpha, \beta > 0$ are parsimony coefficients. Parents are selected via tournament selection, followed by subtree crossover (swapping subtrees between two parents) and subtree mutation (replacing a subtree with a randomly generated valid subtree).

### 3. Architecture & Neural Integration
- **Neural Role:** In the baseline formulation, leaf actions wrap closed-loop trajectory tracking controllers. In neuro-symbolic extensions, neural networks act as learned sensorimotor primitives: (1) convolutional networks estimating 3D bounding boxes and object grasp affords, and (2) goal-conditioned deep RL policies executing localized manipulation trajectories.
- **Interface / Boundary:** Symbolic-to-continuous interface: The discrete GP algorithm searches over tree grammar and topological wiring, treating neural policies as black-box atomic execution leaves that expose standardized boolean return signals.

### 4. Empirical Evaluation & Benchmarks
- **Environments / Tasks:** 
  - Robotic kitting and bin-picking with dynamic obstacles using a dual-arm YuMi robot.
  - Mobile manipulation in Gazebo: navigating to a table, picking up an object, and placing it in a target bin while human agents randomly displace objects.
  - Complex maze navigation under sensor noise and wheel slippage.
- **Baseline Comparisons:**
  - Standard non-reactive Finite State Machines (FSMs).
  - Classical linear automated planning (PDDL / STRIPS open-loop execution).
  - Expert manually designed reactive Behavior Trees.
  - Standard Genetic Programming without disturbance injection during training.
- **Key Quantitative Results:**
  - *Disturbance Resilience:* GP-evolved BTs achieved an $87.5\%$ task completion rate under continuous external human interference, compared to $0.0\%$ for open-loop plans and $42.0\%$ for manually designed BTs (which failed to anticipate unexpected multi-fault compounding states).
  - *Bloat Suppression:* Incorporating parsimony pressure reduced the average evolved tree size from 47 nodes to 11 nodes without degrading success rates, ensuring real-time execution at $100\,\text{Hz}$.

### 5. Failure Modes, Trade-offs & Limitations
- **Interpretability Preservation:** High interpretability: the resulting trees consist entirely of standard Sequence and Fallback control logic and named condition/action primitives. The evolved fallback loops can be directly visualized and audited by safety engineers.
- **Scalability & Gradient Stability:** Combinatorial explosion: GP requires evaluating thousands of tree rollouts in simulation ($\mathcal{O}(N_{\text{pop}} \times N_{\text{gen}} \times K_{\text{trials}})$), creating massive sample complexity bottlenecks ($> 10^5$ environment interactions). Furthermore, genetic crossover frequently produces syntactically redundant or logically contradictory subtrees (e.g., checking conflicting conditions sequentially).
- **Unaddressed Gaps:** Structure search is completely derivative-free and decoupled from continuous policy optimization: continuous control parameters inside action leaves cannot be co-optimized with the tree structure via gradient descent during the evolutionary loop.

