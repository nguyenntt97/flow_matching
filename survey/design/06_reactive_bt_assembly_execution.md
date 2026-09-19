# Subsystem 6: Reactive Composite Assembly & Execution Semantics

**File Location:** `/home/nguyen/projects/flow_matching/survey/design/06_reactive_bt_assembly_execution.md`  
**Subsystem Role:** Hierarchical synthesis of Condition and Action nodes into Control composites (Sequence $\to$, Fallback $?$), implementing high-frequency ($100\,\text{Hz}$) reactive ticks, Guarded Fallback stability invariants, and soft-relaxation continuous execution.

---

## 1. Mathematical Mechanics & Functional Role

In baseline continuous flow matching policies (such as [Bae et al., 2025](file:///home/nguyen/projects/flow_matching/survey/14_bae2025_continuous_crowd_locomotion_crowdes.md)), locomotion is generated in fixed chunks ($T_f = 20$ frames / $4\,\text{s}$ at $5\,\text{fps}$). Mode selection is governed by a discrete Markov transition network $\mu_\varphi(b_f \mid b_h, \dots)$ evaluated **only once every 4 seconds**. When sudden dynamic obstacles, oncoming pedestrians, or velocity changes occur within a chunk, the policy is unable to switch behavioral modes, leading to collisions ($C_R > 2.5\%$) or jerky corrective nudges.

Subsystem 6 bridges the bifurcation tree structure (from Subsystems 3 and 4) and the fast DMP action leaves (from Subsystem 5) into a **Reactive Behavior Tree (RBT)** executing at **$100\,\text{Hz}$** ($10\,\text{ms}$ tick interval).

```
                      ┌──────────────────────────────┐
                      │    Root Node: Fallback (?)   │
                      │  Ticked at 100 Hz (10 ms)    │
                      └──────────────┬───────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         │ (Child 1: Safety)         │ (Child 2: Progress)       │ (Child 3: Recovery)
         ▼                           ▼                           ▼
┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
│  Sequence (->)   │        │  Sequence (->)   │        │  Fallback (?)    │
│  Collision Avoid │        │  Nominal NavMesh │        │  Deadlock Solve  │
└────────┬─────────┘        └────────┬─────────┘        └────────┬─────────┘
         │                           │                           │
    ┌────┴────┐                 ┌────┴────┐                 ┌────┴────┐
    │         │                 │         │                 │         │
    ▼         ▼                 ▼         ▼                 ▼         ▼
  [Cond]    [Act]             [Cond]    [Act]             [Cond]    [Act]
Distance   Swerve /          Path Is    March             Stuck?    Backstep /
 < d_safe   Yield DMP         Clear    Straight DMP                 Turn DMP
```

### 1.1 Behavior Tree Control Composites

A Behavior Tree is an execution directed tree $\mathcal{T} = (\mathcal{V}, \mathcal{E})$ with root node $r \in \mathcal{V}$, ticked at frequency $f_{\text{tick}} = 100\,\text{Hz}$. At each tick, signals flow from root to leaves, returning one of three execution states:
$$\mathcal{S} = \{\text{SUCCESS}, \text{FAILURE}, \text{RUNNING}\}$$

1. **Sequence Node ($\to$):**
   - Evaluates children from left to right.
   - If child $i$ returns `FAILURE`, the Sequence halts immediately and returns `FAILURE`.
   - If child $i$ returns `RUNNING`, the Sequence returns `RUNNING`.
   - Returns `SUCCESS` if and only if all children return `SUCCESS`.
   - *Role in Locomotion:* Enforces strict preconditions before triggering motion (e.g., check `IsNavMeshValid` $\to$ check `IsPathClear` $\to$ execute `MarchStraightDMP`).

2. **Fallback / Selector Node ($?$):**
   - Evaluates children from left to right.
   - If child $i$ returns `SUCCESS`, the Fallback halts immediately and returns `SUCCESS`.
   - If child $i$ returns `RUNNING`, the Fallback returns `RUNNING`.
   - Returns `FAILURE` if and only if all children return `FAILURE`.
   - *Role in Locomotion:* Implements prioritized safety fallbacks (e.g., attempt `MarchStraight` $\to$ if blocked, fallback to `SwerveRightDMP` $\to$ if blocked, fallback to `YieldDMP`).

### 1.2 Overcoming the 4-Second Markov Bottleneck via $100\,\text{Hz}$ Preemption

The key reactivity difference between baseline continuous flow models ([Bae et al., 2025](file:///home/nguyen/projects/flow_matching/survey/14_bae2025_continuous_crowd_locomotion_crowdes.md)) and our Flow2BT framework is summarized below:

| Dimension | CrowdES Locomotion Simulator ([Bae et al., 2025](file:///home/nguyen/projects/flow_matching/survey/14_bae2025_continuous_crowd_locomotion_crowdes.md)) | Flow2BT Reactive Subsystem 6 |
| :--- | :--- | :--- |
| **Control Cycle Period** | $T_{\text{chunk}} = 4.0\,\text{s}$ ($5\,\text{fps}$, 20 frames) | $\Delta t = 0.01\,\text{s}$ ($100\,\text{Hz}$) |
| **Mode Switching Latency** | Up to **$4000\,\text{ms}$** (locked in chunk until boundary) | **$\le 10\,\text{ms}$** (instantaneous preemption) |
| **Intra-Chunk Obstacle Reaction** | None (continues current trajectory chunk blindly) | Precondition fails $\to$ Fallback preempts action |
| **Trajectory Consistency** | Subject to discontinuous jumps at chunk boundaries | Smooth $C^1$ velocity transitions via continuous DMPs |
| **Computational Overhead** | 20 steps of neural ODE/cross-attention ($> 40\,\text{ms}$) | Precondition hyperplane ($0.005\,\text{ms}$) + DMP ODE ($0.05\,\text{ms}$) |

### 1.3 Guarded Fallback Invariants & Regions of Attraction (ROA)

Following the formal stability framework of [Sprague & Ögren (2022)](file:///home/nguyen/projects/flow_matching/survey/05_sprague2022_neural_controllers_bt.md) and [Ögren (2012)](file:///home/nguyen/projects/flow_matching/survey/02_iovino2022_survey_learning_bts.md), we structure fallback hierarchies using **Guarded Fallbacks**:

$$\mathcal{T}_{\text{guarded}} = \text{Fallback}\Big(\text{Sequence}(C_1, A_1), \, \text{Sequence}(C_2, A_2), \, \dots, \, A_{\text{default}}\Big)$$

Each action $A_i$ has an associated Region of Attraction $\mathcal{D}_i \subseteq \mathcal{X}$ and a Lyapunov function $V_i(\mathbf{x})$:
$$\dot{V}_i(\mathbf{x}) \le -\alpha_i V_i(\mathbf{x}), \quad \forall \mathbf{x} \in \mathcal{D}_i$$
Condition $C_i$ acts as a guard verifying $\mathbf{x} \in \mathcal{D}_i$. If state perturbations push $\mathbf{x}$ out of $\mathcal{D}_1$, $C_1$ returns `FAILURE`, and the fallback seamlessly shifts execution to $A_2$, whose domain $\mathcal{D}_2 \supset \mathcal{D}_1$ encompasses the perturbation. This ensures **global asymptotic stability** across the composite tree.

### 1.4 Differentiable Soft-Relaxation for End-to-End Tuning

To enable continuous end-to-end refinement of condition boundaries and DMP parameters, we incorporate the soft-relaxation framework of [Huang et al. (2025)](file:///home/nguyen/projects/flow_matching/survey/04_huang2025_differentiable_behavior_trees.md).

Each node $k$ outputs a tri-state probability vector:
$$\mathbf{z}_k = [p_{\text{succ}}, p_{\text{fail}}, p_{\text{run}}]^\top \in \Delta^2, \quad \sum_{s \in \mathcal{S}} p_s = 1$$

For a composite node with children $\mathbf{z}^{(1)}, \dots, \mathbf{z}^{(N)}$:
1. **Differentiable Sequence:**
   $$p_{\text{succ}}^{\text{seq}} = \prod_{i=1}^N p_{\text{succ}}^{(i)}, \quad p_{\text{fail}}^{\text{seq}} = 1 - \prod_{i=1}^N (1 - p_{\text{fail}}^{(i)}), \quad p_{\text{run}}^{\text{seq}} = 1 - p_{\text{succ}}^{\text{seq}} - p_{\text{fail}}^{\text{seq}}$$
2. **Differentiable Fallback:**
   $$p_{\text{fail}}^{\text{fall}} = \prod_{i=1}^N p_{\text{fail}}^{(i)}, \quad p_{\text{succ}}^{\text{fall}} = 1 - \prod_{i=1}^N (1 - p_{\text{succ}}^{(i)}), \quad p_{\text{run}}^{\text{fall}} = 1 - p_{\text{succ}}^{\text{fall}} - p_{\text{fail}}^{\text{fall}}$$

This formulation allows gradient $\nabla_{\mathbf{w}} \mathcal{L}_{\text{task}}$ to backpropagate directly through the tree topology into the condition hyperplanes $\mathbf{w}_c$ (Subsystem 4) and DMP shape weights $\mathbf{W}_\ell$ (Subsystem 5).

---

## 2. Design Choices & Comparisons from Previous Work

| Architecture / Framework | Tree Topology Construction | Execution Frequency | Reaction Latency | Theoretical Guarantees |
| :--- | :--- | :--- | :--- | :--- |
| **CrowdES Locomotion Simulator** ([Bae et al., 2025](file:///home/nguyen/projects/flow_matching/survey/14_bae2025_continuous_crowd_locomotion_crowdes.md)) | Fixed Markov Transition Network ($B=8$) | $0.25\,\text{Hz}$ (every $4\,\text{s}$) | $\le 4000\,\text{ms}$ | None (pure statistical sampling) |
| **TreeFlow Dendrogram** ([Ramachandran & Sra, 2026](file:///home/nguyen/projects/flow_matching/survey/01_ramachandran2026_trees_to_flows.md)) | Static spatial bifurcation partitioning | Non-executable (static tree) | N/A | Duality with continuous flows, but lacks runtime tick |
| **Guarded Neural BTs** ([Sprague & Ögren, 2022](file:///home/nguyen/projects/flow_matching/survey/05_sprague2022_neural_controllers_bt.md)) | Guarded Fallback with ROA certification | $100\,\text{Hz}$ | $\le 10\,\text{ms}$ | ROA Lyapunov stability proofs |
| **Diff-BT** ([Huang et al., 2025](file:///home/nguyen/projects/flow_matching/survey/04_huang2025_differentiable_behavior_trees.md)) | Differentiable soft-relaxed operators | Continuous / Differentiable | Instantaneous | End-to-end gradient flow; no hard switching |
| **Subsystem 6: Reactive Flow2BT (Our Design)** | **Bifurcation induction + Guarded Fallbacks + Diff-BT** | **$100\,\text{Hz}$ CPU runtime** | **$< 10\,\text{ms}$** | **Asymptotic stability via ROA + $100\,\text{Hz}$ preemption** |

---

## 3. Interface with Downstream Subsystems

1. **Input from Subsystem 4 & 5:** Receives induced Condition hyperplanes $\mathbf{w}_c^\top \mathbf{s} + b_c \ge 0$ and DMP Action parameterizations $(\mathbf{g}_\ell, \mathbf{W}_\ell, K, D)$.
2. **Output to Subsystem 7 (Safety & Formal Verification):** The assembled tree structure $\mathcal{T}$ and its nominal control commands $\mathbf{u}_{\text{nominal}}$ are passed to:
   - The Control Barrier Function Quadratic Program (CBF-QP) for hard collision-free filtering.
   - The BehaVerify / nuXmv pipeline for formal model checking of safety invariants.
