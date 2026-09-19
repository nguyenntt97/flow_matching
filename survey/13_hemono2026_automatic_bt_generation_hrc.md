# Automatic Behavior Tree Generation for Enhanced Human–Robot Collaborative Task Planning in Industry 5.0: A Systematic Review

- **Authors:** Pierre Hémono, Ahmed Nait Chabane, M’hammed Sahnoun, Martin Choux
- **Year / Venue:** 2027 (Available online 2026) / *Robotics and Computer-Integrated Manufacturing* (Elsevier, Vol. 103, Article 103358)
- **Paper Link / Identifier:** [DOI: 10.1016/j.rcim.2026.103358](https://doi.org/10.1016/j.rcim.2026.103358) / [HAL open archive: hal-05662686](https://hal.science/hal-05662686)
- **Primary Category:** Automatic BT Generation & Industrial HRC Survey (Automated Planning, Evolutionary Search, LLMs & Hybrids)

### 1. Executive Summary & Core Hypothesis
In the transition toward Industry 5.0, manufacturing paradigms shift from pure automated efficiency to human-centric, resilient, and sustainable production. Scheduling and coordinating heterogeneous resources—human operators working alongside industrial cobots—demands task allocation frameworks that balance human ergonomics, physical comfort, mental workload, and mutual trust with cycle-time throughput. The authors hypothesize that Behavior Trees (BTs) provide the optimal modular, reactive control architecture for executing complex collaborative action plans, overcoming the structural fragility of Finite State Machines (FSMs) and the static execution limits of Hierarchical Task Networks (HTNs). This systematic review synthesizes recent breakthroughs in the automated generation of BTs, categorizing synthesis methodologies across classical automated planning, evolutionary metaheuristics, reinforcement learning, and emerging Large Language Model (LLM) agents within collaborative industrial assembly and scheduling.

### 2. Theoretical Framework & Mathematical Formulation
- **Node Mechanics & Return Status Handling:** Classical reactive Behavior Tree execution semantics propagating $S \in \{\text{SUCCESS}, \text{FAILURE}, \text{RUNNING}\}$. Leaf nodes are categorized into: (1) condition nodes inspecting workspace state, human safety zones, and part readiness; (2) robotic action nodes commanding cobot manipulators; and (3) collaborative interaction nodes coordinating handovers and shared workstation access. The reactive tick frequency ($10 - 100\,\text{Hz}$) ensures real-time preemption when human operators enter safety-monitored envelopes.
- **Execution / Control Flow:** The survey categorizes task representations used across industrial scheduling into:
  1. *Production Problem Formulations:* Job Shop Scheduling Problem (JSSP) and Assembly Line Balancing Problem (ALBP) adapted for human-robot shared cells:
     $$\min \quad \alpha \cdot C_{\max} + \beta \cdot \sum_{i} \text{ErgoScore}(i) + \gamma \cdot \text{IdleTime}$$
     where $C_{\max}$ is the makespan and $\text{ErgoScore}$ penalizes poor human postures and repetitive fatigue.
  2. *Control Architecture Trade-Offs:*
     - *FSMs:* State-space explosion $\mathcal{O}(2^{|S|})$ and tight coupling between states prevent dynamic insertion of collaborative recovery routines.
     - *HTNs:* Strong hierarchical decomposition but historically lack fast reactive preemption when humans deviate from nominal assembly sequences.
     - *BTs:* Decoupled modular subtrees; Sequence $(\to)$ handles multi-step assembly, Fallback $(?)$ handles human task assistance or emergency halts, and Parallel $(\rightrightarrows)$ coordinates concurrent human-cobot tasks.
- **Optimization & Generation Taxonomy:** The authors systematize automatic BT generation techniques into five core paradigms:
  1. *Automated Planning to BT (Model-Based):* Converting PDDL / STRIPS domain formulations into BTs via backward-chaining (e.g. expanding preconditions and effects into Fallback-Sequence branches).
  2. *Evolutionary & Metaheuristic Algorithms:* Genetic Programming (GP) and Grammatical Evolution (GE) searching syntax trees against multi-objective fitness functions balancing makespan, ergonomics, and tree complexity.
  3. *Reinforcement & Imitation Learning:* Discovering subtrees from human demonstration logs or RL policies, mapping Options to reactive subtrees.
  4. *Large Language Models (LLMs) & Foundation Models:* Translating natural language instructions and procedural assembly manuals into executable BT domain-specific languages (e.g., XML/BehaviorTree.CPP) via in-context learning and retrieval-augmented verification.
  5. *Hybrid Methods:* Combining high-level symbolic planners for global constraint satisfaction with reactive BT generation for local disturbance management.

### 3. Architecture & Neural Integration
- **Neural Role:** In modern AI-driven BT generation pipelines, neural networks operate across two distinct tiers:
  1. *Generative Synthesis Tier:* Pretrained LLMs and vision-language models (VLMs) acting as high-level planners that parse multi-modal human intent, assembly work instructions, and scene descriptions into candidate Behavior Tree topologies.
  2. *Low-Level Execution & Perception Tier:* Vision models estimating human 3D skeleton poses, tracking part locations, and computing real-time ergonomic risk scores (e.g., REBA/RULA), directly informing BT condition leaves.
- **Interface / Boundary:** Separation of planning time and execution time: high-level neural/symbolic generators emit structured BT graphs (JSON, XML, or py_trees code) that are validated through syntax linters or model checkers before being passed to deterministic, reactive execution engines.

### 4. Empirical Evaluation & Industrial Applications
- **Environments / Tasks:** 
  - Collaborative robotic assembly lines (automotive sub-assemblies, aerospace riveting, electronics packaging).
  - Human-robot cooperative kitting, bin picking, and shared workstation part feeding.
  - Multi-objective Job Shop Scheduling (JSSP) and Assembly Line Balancing (ALBP) benchmarks.
- **Comparative Findings on Generation Methodologies:**
  - *Planning-based (PDDL $\to$ BT):* Guarantees logical correctness and goal reachability if the domain is accurately modeled, but suffers in unstructured or partially observable environments.
  - *Evolutionary (GP / GE):* Excellent at discovering unexpected fault-recovery behaviors, but sample complexity and high simulation overhead limit online re-synthesis during live shifts.
  - *LLM-Assisted Generation:* Radically lowers engineering barriers by enabling operators to prompt cobots with natural language; however, hallucinations, lack of formal safety proofs, and prompt sensitivity necessitate deterministic grammar wrappers and syntax filters.
- **Human-Centric Factors in Industry 5.0:**
  - *Ergonomics & Workload:* Incorporating physical load monitoring directly into task allocation trees improves worker satisfaction and reduces musculoskeletal injuries.
  - *Operator Trust & Legibility:* The modular, human-interpretable nature of Behavior Trees enables operators to comprehend cobot intentions and intervene safely, unlike opaque end-to-end black-box controllers.

### 5. Failure Modes, Trade-offs & Limitations
- **Interpretability vs. Expressivity:** While hand-crafted or grammar-constrained BTs maintain transparent readability, automatically synthesized trees (especially from evolutionary search or unconstrained LLMs) can suffer from structural bloat, redundant condition checks, and obscure nesting that diminish human auditability.
- **Online Adaptation Latency:** When human operators exhibit non-deterministic deviations from planned workflows, full tree re-synthesis can cause unacceptable delays on live production lines. Localized reactive fallback subtrees and parameterized execution nodes are necessary to maintain continuous throughput.
- **Formal Verification Gap in Generative Methods:** Data-driven and LLM-generated Behavior Trees lack formal correctness guarantees. Integrating automatic BT synthesis with symbolic model checkers (such as nuXmv / BehaVerify) and Control Barrier Functions is identified as a vital prerequisite for safety-critical industrial deployment.

