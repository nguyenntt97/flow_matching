# Subsystem 3: Topological Induction & Bifurcation Analysis

**File Location:** `/home/nguyen/projects/flow_matching/survey/design/03_topological_induction_bifurcations.md`  
**Subsystem Role:** Discovering the macro-tree topology from continuous trajectory bundles via reverse-time flow coarse-graining, eliminating combinatorial tree structure search.

---

## 1. Mathematical Mechanics & Functional Role

A central bottleneck in Behavior Tree synthesis has historically been **topology discovery**: how to structure the hierarchy of Sequence and Fallback control nodes without testing millions of random candidate trees.

Subsystem 3 leverages the **Tree-Flow Duality** established by [Ramachandran & Sra (2026)](../01_ramachandran2026_trees_to_flows.md): running continuous flow trajectories backward in time mathematically reverses the entropy production of decision-making, merging localized multimodal streamlines into hierarchical bifurcation trees (dendrograms).

```
Continuous Trajectory Bundle Ξ = {ξ_i(t)}
                   │
                   ▼  Reverse-Time Agglomerative Clustering (τ = 1 - t)
Pairwise Spectral Trajectory Distance Matrix: D(ξ_i, ξ_j)
                   │
                   ▼  Kramers-Moyal Coarse-Graining
Hierarchical Topological Dendrogram: T_topo
                   │
                   ├──────────────────────────────────┐
                   ▼                                  ▼
      [ Bifurcation Node k ]               [ Terminal Leaf Partition ℓ ]
   Identifies where paths split:           Unimodal, low-curvature trajectory
   • "Pass Obstacle Left"                  bundle ready for DMP fitting
   • "Pass Obstacle Right"                 (Subsystem 5)
   • "Yield / Stop"
         │
         ▼ Passed to Subsystem 4 (Condition Induction)
```

### 1.1 Pairwise Trajectory Metric
Given two continuous rollouts $\xi_i(t), \xi_j(t) \in \mathbb{R}^2$ from Subsystem 2 over duration $t \in [0, T_f]$:
$$D(\xi_i, \xi_j) = \int_0^{T_f} \|\xi_i(t) - \xi_j(t)\|_2^2 \, dt + \lambda_{\text{term}} \|\xi_i(T_f) - \xi_j(T_f)\|_2^2 + \lambda_{\text{vel}} \int_0^{T_f} \|\dot{\xi}_i(t) - \dot{\xi}_j(t)\|_2^2 \, dt$$
* The first term measures spatial deviation across the path.
* The terminal term $\lambda_{\text{term}}$ heavily penalizes arriving at different destinations.
* The velocity term $\lambda_{\text{vel}}$ separates stopping/yielding behaviors from continuous walking even if spatial paths overlap.

### 1.2 Reverse-Time Coarse-Graining & Dendrogram Extraction
Under the continuum limit of hierarchical partitions proved by Ramachandran & Sra:
1. At $\tau = 0$ (forward data distribution at destination), trajectories end in distinct spatial clusters.
2. As $\tau \to 1$ (tracing backward in time to initial encounter states), the distinct modes merge as mutual information decreases.
3. Performing **Ward's minimum variance agglomerative clustering** on matrix $\mathbf{D} = [D(\xi_i, \xi_j)]$ builds a binary hierarchical tree $\mathcal{T}_{\text{topo}}$.
4. **Bifurcation Point Detection:** A node $k \in \mathcal{T}_{\text{topo}}$ is marked as a critical bifurcation if the within-cluster variance increases discontinuously ($\Delta \text{Var} > \epsilon_{\text{split}}$). This corresponds to a critical decision point (e.g. deciding whether to veer left or right to avoid an oncoming pedestrian).

---

## 2. Design Choices & Comparisons from Previous Work

| Approach | Topology Discovery Mechanism | Sample Complexity | Tree Quality | Why Chosen / Adapted in Flow2BT |
| :--- | :--- | :--- | :--- | :--- |
| **Genetic Programming (GP-BT)** ([Iovino et al., 2021](../08_iovino2021_gp_bt_unpredictable.md)) | Random subtree crossover & mutation over syntax trees | Extremely high ($> 10^5$ rollouts) | Prone to structural bloat and redundant subtrees | Rejected: sample complexity is intractable for complex crowds |
| **Monte Carlo DAG Search (MCDAGS)** ([Scheide et al., 2021](../09_scheide2021_mcdags_grammar_bt.md)) | MCTS search over formal context-free grammar | High ($> 10^4$ simulation iterations) | Requires hand-crafted grammar rules | Rejected: rigid grammar definitions limit adaptation |
| **K-Means on Trajectory Chunks** ([Bae et al., 2025](../14_bae2025_continuous_crowd_locomotion_crowdes.md)) | Flat K-means ($B=8$ clusters) on normalized vectors | Very fast ($< 1$ second) | Flat clusters; **no hierarchical tree structure** | **Adapted**: We retain their $B=8$ behavioral concept, but replace flat K-means with hierarchical agglomerative clustering |
| **Flow-to-Tree Duality (Dendrogram Extraction)** ([Ramachandran & Sra, 2026](../01_ramachandran2026_trees_to_flows.md)) | Reverse-time coarse-graining of flow trajectories | **Zero environment interactions** (uses offline rollouts) | Clean, non-redundant, mathematically grounded hierarchy | **Adopted as Subsystem 3**: Discovers the macro-BT skeleton directly from the continuous flow teacher |

---

## 3. Interface with Downstream Subsystems

1. **To Subsystem 4 (Condition Induction):** Provides the split indices at each bifurcation node $k$. Each bifurcation defines two trajectory subsets:
   $$\Xi_{\text{left}}^{(k)} \quad \text{vs.} \quad \Xi_{\text{right}}^{(k)}$$
   which serve as positive and negative training classes for the Condition Node classifier.
2. **To Subsystem 5 (DMP Action Leaves):** Provides the terminal leaf clusters $\Xi_\ell$. Each leaf cluster $\ell$ is guaranteed to be unimodal and low-curvature, providing clean trajectory bundles for fitting individual DMPs.
3. **To Subsystem 6 (Reactive BT Assembly):** Defines the hierarchical parent-child relationships and identifies alternative branches for Fallback ($?$) node composition.

