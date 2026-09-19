# Subsystem 1: Global Spatial Layout, Traversability & NavMesh Guidance

**File Location:** `/home/nguyen/projects/flow_matching/survey/design/01_global_guidance_navmesh.md`  
**Subsystem Role:** Environmental perception, global topology extraction, non-convex obstacle representation, and waypointing for continuous pedestrian locomotion.

---

## 1. Mathematical Mechanics & Functional Role

In continuous pedestrian locomotion, agents navigate across non-convex spatial layouts (corridors, intersections, plazas with trees, pillars, and urban structures). A local reactive controller or flow policy cannot plan long-term global paths across U-shaped obstacles without suffering from local minima. 

Subsystem 1 provides the **global topological backbone** that seeds continuous flow trajectories with reachable goals and waypoints:

```
[ Input Scene Image I / Geometric Map ]
                   │
                   ▼  Semantic Segmentation & Masking
[ Binary Traversable Map M_W = 1 - (Obstacles + Structures) ]
                   │
                   ▼  Recast Polygon Triangulation
[ Navigation Mesh (NavMesh) : Connected Convex Polygons {P_k} ]
                   │
                   ▼  A* Search on Polygon Adjacency Dual Graph
[ Global Polyline Path : Gamma = [c_0, w_1, w_2, ..., c_{T_d}] ]
                   │
                   ▼  Sliding Horizon Pace Projection (nu)
[ Local Continuous Attractor Waypoint : c_{t, nav} ]
```

### 1.1 Geometry & Traversability Formulation
1. **Traversable Map Construction:**  
   Given a top-down aerial or perspective scene $\mathcal{I}$, an environmental segmentation model (e.g. Grounded-SAM / SegFormer as in [Bae et al., 2025](file:///home/nguyen/projects/flow_matching/survey/14_bae2025_continuous_crowd_locomotion_crowdes.md)) identifies non-traversable semantic classes $\mathcal{C}_{\text{obs}} = \{\text{buildings}, \text{structures}, \text{bushes}, \text{water}, \text{fences}\}$. The binary traversable space is:
   $$\mathcal{M}_W(\mathbf{p}) = \begin{cases} 1 & \text{if } \text{Class}(\mathbf{p}) \notin \mathcal{C}_{\text{obs}} \\ 0 & \text{otherwise} \end{cases}, \quad \mathbf{p} \in \mathbb{R}^2$$

2. **Navigation Mesh (NavMesh) Discretization:**  
   Rather than using dense raster grids which scale poorly, the continuous traversable space $\mathcal{M}_W$ is decomposed into a set of connected 2D convex polygons using the Recast polygonization algorithm:
   $$\Omega_{\text{nav}} = \bigcup_{k=1}^K \mathcal{P}_k, \quad \mathcal{P}_i \cap \mathcal{P}_j = \text{Shared Portal Edge } e_{ij}$$
   Because each polygon $\mathcal{P}_k$ is convex, any two points within the same polygon can be connected via a straight line without collision.

3. **A* Dual-Graph Pathfinding & Funnel Smoothing:**  
   Given current position $\mathbf{c}_t \in \mathcal{P}_{\text{start}}$ and destination $\mathbf{c}_{T_d} \in \mathcal{P}_{\text{goal}}$, A* search over the polygon adjacency graph finds the shortest portal sequence, and the Simple Stupid Funnel Algorithm extracts the shortest Euclidean polyline:
   $$\Gamma(\mathbf{c}_t, \mathbf{c}_{T_d}) = [\mathbf{c}_t, \mathbf{w}_1, \mathbf{w}_2, \dots, \mathbf{c}_{T_d}]$$

4. **Dynamic Waypoint Projection:**  
   To supply the continuous flow model and low-level motor primitives with a dynamic target, the local guidance waypoint $\mathbf{c}_{t, \text{nav}}$ is extracted at distance equal to the agent's nominal walking pace $\nu$ along $\Gamma$:
   $$\mathbf{c}_{t, \text{nav}} = \Gamma\left(\text{ArcLength}(\mathbf{c}_t) + \nu \cdot \Delta t_{\text{horizon}}\right)$$

---

## 2. Design Choices & Comparisons from Previous Work

| Approach | Representation | Strengths | Weaknesses | Why Chosen / Adapted in Flow2BT |
| :--- | :--- | :--- | :--- | :--- |
| **Grid-Based A\* / Dijkstra** (Classical) | 2D Occupancy Grid ($0.1\,\text{m}$ cells) | Exact, easy to implement | High memory footprint; produces jerky, grid-aligned paths | Rejected for global planning; retained only for local laser costmaps |
| **Pure End-to-End Neural Guidance** (Direct Diffusion / RL) | Monolithic CNN/Transformer mapping images $\to$ actions | Learns visual affordances directly | Gets trapped in non-convex U-shaped dead ends; lacks reachability proofs | Rejected: unsafe and computationally expensive |
| **NavMesh Polyline Waypoints** ([Bae et al., 2025](file:///home/nguyen/projects/flow_matching/survey/14_bae2025_continuous_crowd_locomotion_crowdes.md)) | Convex polygonal mesh + A* Funnel polyline | Handles arbitrary complex maps; computationally lightweight ($< 1\,\text{ms}$) | Static; does not account for dynamic crowds on its own | **Adopted as Subsystem 1**: Provides the nominal attractor goal $\mathbf{g} = \mathbf{c}_{t, \text{nav}}$ for the continuous flow leaves |
| **Sampling-Based RRT\*** | Random tree expansion in $\mathbb{R}^2$ | Probabilistically complete | Non-deterministic, path variance across ticks | Rejected: non-repeatable attractor goals destabilize DMP fitting |

---

## 3. Interface with Downstream Subsystems

Subsystem 1 exposes a clean functional interface to the rest of the Learnable Behavior Tree:

1. **To Subsystem 2 (Teacher Flow Policy):** Provides the conditioning waypoint $\mathbf{c}_{t, \text{nav}}$ and local traversability mask $\mathcal{M}_W$, ensuring that sampled trajectories flow along valid corridors.
2. **To Subsystem 4 (Condition Nodes):** Provides distance-to-waypoint and corridor clearance signals (`IsWaypointReachable?`, `IsNavMeshValid?`) that serve as boolean condition checks in Sequence nodes.
3. **To Subsystem 5 (DMP Action Leaves):** Directly sets the attractor goal position:
   $$\mathbf{g}_\ell = \mathbf{c}_{t, \text{nav}}$$
   allowing the DMP second-order dynamics to pull the pedestrian smoothly toward the NavMesh polyline.
