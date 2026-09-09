# Behavior Tree Learning for Robotic Task Planning through Monte Carlo DAG Search over a Formal Grammar

- **Authors:** Emily Scheide, Graeme Best, Geoffrey A. Hollinger
- **Year / Venue:** 2021 / IEEE Robotics and Automation Letters (RA-L) with ICRA 2021 presentation
- **Paper Link / Identifier:** [IEEE Xplore / DOI: 10.1109/LRA.2021.3062335](https://doi.org/10.1109/LRA.2021.3062335)
- **Primary Category:** Structure Evolution + Policy Gradient

### 1. Executive Summary & Core Hypothesis
Synthesizing Behavior Trees from demonstration or reward requires exploring a combinatorial space of tree topologies. Standard evolutionary methods (like GP) suffer from bloat, duplicate tree evaluations, and lack of systematic lookahead. Scheide, Best, and Hollinger hypothesize that Behavior Tree synthesis can be formulated as a formal grammar traversal problem solved via Monte Carlo Directed Acyclic Graph Search (MCDAGS). By representing grammar derivations as a DAG rather than a tree—enabling identical subtrees produced by different derivation paths to share search statistics—and integrating simulated annealing into tree selection, the algorithm discovers high-performing Behavior Trees with significantly reduced computational search effort.

### 2. Theoretical Framework & Mathematical Formulation
- **Node Mechanics & Return Status Handling:** Standard discrete Behavior Tree return statuses $\{\text{SUCCESS}, \text{FAILURE}, \text{RUNNING}\}$. Leaf actions wrap robotic motor primitives (e.g., NavigateTo, InspectTarget, SurfaceAndCommunicate), while conditions evaluate environmental sensor readings (e.g., BatteryLow, TargetDetected).
- **Execution / Control Flow:** A formal context-free grammar (CFG) defines the valid search space of BT topologies: $G = (\mathcal{V}_N, \mathcal{V}_T, \mathcal{P}, S)$, where:
  - $\mathcal{V}_N = \{BT, Seq, Fall, Action, Condition\}$ are non-terminal symbols.
  - $\mathcal{V}_T$ represents composite operators ($\to, ?$) and atomic action/condition leaves.
  - $\mathcal{P}$ contains production rules (e.g. $BT \to \text{Seq}(BT, BT) \mid \text{Fall}(BT, BT) \mid Action$).
  Every valid Behavior Tree corresponds to a complete derivation from start symbol $S$. Because multiple derivation sequences yield semantically and syntactically identical Behavior Trees, the derivation state space is represented as a Directed Acyclic Graph (DAG) $\mathcal{D} = (\mathcal{V}_D, \mathcal{E}_D)$, where nodes represent partially expanded derivation sentential forms, and edges represent production rule expansions.
- **Optimization Strategy:** The algorithm extends Monte Carlo Tree Search (MCTS) to DAGs (MCDAGS) across four iterative phases:
  1. *Selection:* Traverses the DAG from root $S$ using a modified Upper Confidence Bound for Trees (UCT) adapted for DAG multi-parent structures:
     $$\text{UCT}(u, v) = \bar{X}(v) + 2 C_p \sqrt{\frac{2 \ln N(u)}{N(v)}}$$
     where $\bar{X}(v)$ is the aggregated reward backpropagated through all parent paths leading to $v$, and $N(v)$ is the visit count.
  2. *Expansion:* When a non-terminal leaf in the search DAG is reached, one or more valid grammar production rules from $\mathcal{P}$ are applied. If the resulting sentential form has been visited via an alternative path, edges are merged, preserving DAG compactness.
  3. *Rollout (Simulation):* Unexpanded non-terminals are expanded uniformly at random or guided by simulated annealing until a complete terminal Behavior Tree is generated. The resulting BT is evaluated in a robotic simulation environment to obtain episodic return $R$.
  4. *Backpropagation:* The reward $R$ is backpropagated up the DAG across all incoming ancestor edges, updating visit counts and value estimates along all directed paths.

### 3. Architecture & Neural Integration
- **Neural Role:** In the primary formulation, MCDAGS searches over symbolic action and condition leaves. In neuro-symbolic extensions, neural network models serve as: (1) learned transition dynamics models used for fast internal forward simulation rollouts during the MCTS rollout phase, and (2) neural perceptual classifiers embedded inside condition nodes evaluating high-dimensional sensor data (e.g. sonar images, camera feeds).
- **Interface / Boundary:** Symbolic grammar boundary: MCDAGS governs the discrete topological assembly of the tree, while continuous sensor inputs and low-level PID/trajectory controllers are encapsulated inside the terminal leaf nodes.

### 4. Empirical Evaluation & Benchmarks
- **Environments / Tasks:** 
  - Autonomous Underwater Vehicle (AUV) marine exploration and target search under communication blackout and battery depletion constraints.
  - Long-horizon multi-target search-and-rescue navigation with dynamic hazard zones.
  - Abstract task planning benchmark comparing combinatorial coverage against baseline grammar search methods.
- **Baseline Comparisons:**
  - Standard Genetic Programming Behavior Trees (GP-BT).
  - Standard Monte Carlo Tree Search over tree derivations (without DAG node merging).
  - Random Grammar Sampling with Hill Climbing.
  - Manually engineered baseline Behavior Trees.
- **Key Quantitative Results:**
  - *Search Efficiency:* MCDAGS discovered optimal mission Behavior Trees using $62\%$ fewer environment evaluations than standard tree-based MCTS and $4.8\times$ fewer evaluations than GP-BT, directly demonstrating the efficiency of DAG multi-parent value aggregation.
  - *Mission Return:* In the AUV target search domain, MCDAGS achieved an average mission score of $92.4 \pm 3.1$, matching hand-crafted expert trees ($91.8 \pm 2.8$) and outperforming GP-BT ($78.6 \pm 6.4$) by effectively avoiding deadlocks during battery depletion events.

### 5. Failure Modes, Trade-offs & Limitations
- **Interpretability Preservation:** Outstanding interpretability: every candidate generated by the formal grammar is a valid, well-formed Behavior Tree adhering strictly to expert-defined syntactic constraints. The resulting trees contain no dead branches or invalid operator sequences.
- **Scalability & Gradient Stability:** Scalability is limited by grammar branching factor. As the number of available action and condition primitives expands ($|\mathcal{V}_T| > 50$), the DAG search space grows exponentially, requiring significant rollout sampling or heuristic grammar pruning to prevent search timeouts.
- **Unaddressed Gaps:** Discrete parameters only: continuous parameters (such as navigation target coordinates, velocity limits, or sensor thresholds) must be discretized into a finite set of grammar tokens prior to search; continuous gradient-based parameter tuning is not integrated into the DAG search loop.

