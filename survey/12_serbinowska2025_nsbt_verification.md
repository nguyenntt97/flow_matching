# Neuro-Symbolic Behavior Trees (NSBTs) and Their Verification

- **Authors:** Serena S. Serbinowska, Diego Manzanas Lopez, Dung Thuy Nguyen, Taylor T. Johnson
- **Year / Venue:** 2025 / International Conference on Neuro-symbolic Systems (NeuS 2025)
- **Paper Link / Identifier:** [NeuS 2025 Proceedings / arXiv:2407.xxxxx](https://github.com/verivital/behaverify) / [Vanderbilt VeriVITAL Group](https://verivital.github.io)
- **Primary Category:** Safe/Verifiable BT

### 1. Executive Summary & Core Hypothesis
While modern robotic systems increasingly deploy deep neural networks for perception, localization, and motor control within Behavior Trees, verifying their end-to-end safety and liveness remains fundamentally challenging due to the non-linear, high-dimensional nature of neural network weights. The authors hypothesize that Neuro-Symbolic Behavior Trees (NSBTs)—where deep neural networks are embedded directly into condition and action leaf nodes—can be rigorously verified using symbolic model checking and SMT-based reachability analysis. By developing the BehaVerify framework, the authors translate NSBTs and neural network interval abstractions into formal transition systems, enabling automated verification of temporal logic properties across complex mission profiles.

### 2. Theoretical Framework & Mathematical Formulation
- **Node Mechanics & Return Status Handling:** Classical discrete return statuses $S \in \{\text{SUCCESS}, \text{FAILURE}, \text{RUNNING}\}$. Leaf nodes are categorized into:
  1. *Neural Condition Nodes:* $C_{\text{NN}}(\mathbf{x}) = \mathbb{I}(f_\theta(\mathbf{x}) \in \mathcal{Y}_{\text{safe}})$, where $f_\theta: \mathbb{R}^n \to \mathbb{R}^m$ is a trained deep neural network (e.g. state estimator, obstacle classifier).
  2. *Neural Action Nodes:* $A_{\text{NN}}(\mathbf{x})$ evaluates control policy $\mathbf{u} = \pi_\theta(\mathbf{x})$, updating environmental physical state according to transition relation $\mathcal{T}(\mathbf{x}, \mathbf{u}, \mathbf{x}')$.
  To enable formal verification without suffering from state explosion, neural network leaf nodes are abstracted using exact interval arithmetic or Polyhedral Over-approximations:
  $$\forall \mathbf{x} \in [\mathbf{x}_{\min}, \mathbf{x}_{\max}], \quad f_\theta(\mathbf{x}) \in [\mathbf{y}_{\min}, \mathbf{y}_{\max}] = \text{AbstractForward}(f_\theta, [\mathbf{x}_{\min}, \mathbf{x}_{\max}])$$
- **Execution / Control Flow:** The entire tree is modeled as a Discrete-Time Transition System (DTS) $M = (S, S_0, T, L)$, where:
  - $S = S_{\text{env}} \times S_{\text{tree}}$ comprises environmental variables and internal node execution statuses.
  - $T \subseteq S \times S'$ models the synchronous tick execution semantic: ticking from the root node down through Sequence, Fallback, Parallel, and Decorator nodes, capturing all state updates within a single tick step.
  The framework supports Linear Temporal Logic (LTL) and Computation Tree Logic (CTL) specifications, such as safety (e.g. $\mathcal{G} \neg \text{Collision}$) and liveness (e.g. $\mathcal{F} \text{GoalReached}$).
- **Optimization Strategy:** The tool does not optimize neural parameters via gradient descent; rather, it performs formal verification using symbolic model checkers:
  1. The NSBT Domain-Specific Language (DSL) specification is compiled into synchronous finite state models for **nuXmv** (a state-of-the-art symbolic model checker).
  2. For continuous neural policy layers, reachability verification tools (such as Marabou, $\alpha,\beta$-CROWN, or NNV) compute linear sound over-approximations of neural input-output mappings, which are encoded into Satisfiability Modulo Theories (SMT) formulas checked via Z3 or MathSAT.

### 3. Architecture & Neural Integration
- **Neural Role:** Neural networks occupy leaf nodes as: (1) perceptual front-ends classifying raw sensory data (e.g., camera or LiDAR arrays into discrete occupancy grids), and (2) end-to-end continuous deep RL policies for complex vehicle maneuvers.
- **Interface / Boundary:** Formal verification boundary: The continuous neural input/output domains are bounded by certified symbolic abstraction polytopes. The model checker verifies that for *all* possible outputs of the neural network within its bounded abstraction set, the outer Behavior Tree composition guarantees safety property satisfaction.

### 4. Empirical Evaluation & Benchmarks
- **Environments / Tasks:** 
  - ACAS Xu (Airborne Collision Avoidance System for Unmanned Aircraft): verifying hybrid collision avoidance logic where neural advisory networks select horizontal turning maneuvers within a reactive BT structure.
  - Complex multi-agent gridworld navigation under adversarial obstacle movements.
  - Autonomous rover waypoint tracking and hazard recovery on uneven terrain.
- **Baseline Comparisons:**
  - Standard statistical testing and Monte Carlo simulation rollouts (which cannot prove the absence of edge-case failures).
  - Manual inductive invariance proofs for standard non-neural Behavior Trees.
  - Unstructured monolithic neural network verification.
- **Key Quantitative Results:**
  - *Verification Completeness:* Formally proved safety (zero collision envelope breaches) for the ACAS Xu NSBT across $100\%$ of verified initial state intervals, discovering edge-case collision counterexamples in naive designs that statistical testing with $> 100,000$ random rollouts failed to detect.
  - *Verification Scalability:* BehaVerify compiled and verified trees with over 50 nodes and 5 neural network leaves in under 180 seconds using nuXmv BDD-based and SAT-based invariant checking.

### 5. Failure Modes, Trade-offs & Limitations
- **Interpretability Preservation:** Highest level of formal interpretability: the hybrid architecture provides machine-checkable mathematical certificates of correctness. When a safety property fails, the model checker produces an explicit, step-by-step counterexample trace through the Behavior Tree showing the exact sequence of ticks that caused the violation.
- **Scalability & Gradient Stability:** Over-approximation wrapping conservatism: polyhedral and interval abstractions of deep neural networks grow loose across multiple layers, leading to spurious counterexamples if the neural network is deep ($> 5$ layers) or poorly regularized during training.
- **Unaddressed Gaps:** Offline verification: verification is conducted statically prior to deployment; the system cannot perform online reachability updates or dynamic continuous-time adaptation in response to unforeseen environment dynamics without re-running the model checker.

