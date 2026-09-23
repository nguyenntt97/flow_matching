# Subsystem 2: Teacher Continuous Flow Policy & Trajectory Ensemble Generation

**File Location:** `/home/nguyen/projects/flow_matching/survey/design/02_teacher_flow_policy.md`  
**Subsystem Role:** Learning continuous multimodal pedestrian velocity fields, conditioning on neighboring pedestrians and local NavMesh goals, and generating offline trajectory ensembles for tree distillation.

---

## 1. Mathematical Mechanics & Functional Role

In the Flow2BT architecture, the **Teacher Flow Policy** acts as an exploratory expert. It is trained on real-world pedestrian demonstration datasets (e.g. ETH, UCY, Stanford Drone Dataset) to model the continuous probability distribution over footstep velocities conditioned on social interactions and local geometry:

```
[ Local State s_t = (c_t, v_t, c_{t, nav}) ] ──┐
[ Neighbor Trajectories H_neighbors ] ─────────┼──► [ Multi-Agent Interaction Encoder ]
[ Local Traversability Grid M_W (16x16m) ] ───┘             (Graph / Cross-Attention)
                                                                       │
                                              Context Vector C_s ◄─────┘
                                                              │
               Gaussian Noise x_0 ~ N(0, I) ──┐               │
               Time Step t ~ U[0, 1] ─────────┼──► [ Continuous Flow Matching Vector Field ]
               Interpolated Velocity x_t ─────┘            v_theta(x_t, t, C_s)
                                                              │
                                                              ▼
                                            Predicted Target Velocity: dx_t / dt
```

### 1.1 Flow Matching Formulation for Locomotion
Using the core library's [`AffineProbPath`](../../flow_matching/path/affine.py#L15-L261) and [`CondOTScheduler`](../../flow_matching/path/scheduler/scheduler.py#L104):
1. **Target Distribution ($p_1$):** Ground-truth pedestrian velocities $\mathbf{v}_1 = \dot{\mathbf{c}} \in \mathbb{R}^2$ or footstep displacements over horizon $T_f$.
2. **Prior Distribution ($p_0$):** Standard Gaussian noise $\mathbf{v}_0 \sim \mathcal{N}(0, \mathbf{I})$.
3. **Conditional Linear Trajectory:**
   $$\mathbf{v}_t = (1 - t) \mathbf{v}_0 + t \mathbf{v}_1, \quad \dot{\mathbf{v}}_t = \mathbf{v}_1 - \mathbf{v}_0$$
4. **Conditional Flow Matching Loss:**
   $$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0, 1], \, \mathbf{v}_1 \sim p_{\text{data}}, \, \mathbf{v}_0 \sim p_0, \, \mathbf{C}_s} \left[ \left\| v_\theta(\mathbf{v}_t, t, \mathbf{C}_s) - (\mathbf{v}_1 - \mathbf{v}_0) \right\|_2^2 \right]$$

### 1.2 Conditioning Context ($\mathbf{C}_s$)
Following the locomotion simulator inputs from [Bae et al. (2025)](../14_bae2025_continuous_crowd_locomotion_crowdes.md):
$$\mathbf{C}_s = \left[ \mathbf{c}_{t, \text{nav}} - \mathbf{c}_t, \, \nu, \, \text{SocialGraph}(\mathcal{H}_{\text{neighbors}}), \, \text{CNN}(\mathcal{M}_W) \right]$$
* **NavMesh Relative Goal Vector:** $\mathbf{c}_{t, \text{nav}} - \mathbf{c}_t$ guides nominal heading.
* **Social Graph Attention:** Graph Neural Network or Transformer attention pooling over the relative displacement and velocity vectors of all pedestrians within a $16 \times 16\,\text{m}$ perceptual window:
  $$\mathbf{h}_{\text{social}} = \sum_{j \in \mathcal{N}(i)} \text{Softmax}\left(\frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d}}\right) \mathbf{v}_j$$

### 1.3 Trajectory Ensemble Rollouts
To prepare the dataset for topological tree extraction (Subsystem 3), the teacher model is rolled out across an ensemble of $M$ diverse initializations using [`ODESolver`](../../flow_matching/solver/ode_solver.py#L17-L204):
$$\Xi = \{\xi_i(\tau)\}_{i=1}^M, \quad \dot{\xi}_i(\tau) = v_\theta(\xi_i(\tau), \tau, \mathbf{C}_s^{(i)}), \quad \tau \in [0, T_f]$$
This generates a rich bundle of continuous trajectories capturing all navigation strategies (e.g. diverging left around an obstacle vs. swerving right vs. slowing down).

---

## 2. Design Choices & Comparisons from Previous Work

| Approach | Model Type | Inference Latency | Handling of Multimodality | Why Chosen / Adapted in Flow2BT |
| :--- | :--- | :--- | :--- | :--- |
| **Deterministic Regression** (Social-LSTM, GCNs) | Single forward-pass MLP/GRU | Ultra-fast ($< 2\,\text{ms}$) | **Fails**: Averages conflicting paths (leads to collisions in bottlenecks) | Rejected: cannot capture multimodal branching |
| **DDIM Diffusion Policy** ([Bae et al., 2025](../14_bae2025_continuous_crowd_locomotion_crowdes.md)) | Iterative Gaussian denoising (50 steps) | Slow ($\approx 50 - 100\,\text{ms}$ per agent) | High diversity; covers multimodal evasion routes | Valuable as an offline teacher, but too slow for online control |
| **Conditional Flow Matching (CFM)** ([Lipman et al., 2024](../../README.md)) | Straight-path ODE vector field | Medium ($5 - 10$ steps via Heun/Dopri5) | High diversity, lower curvature than diffusion | **Adopted as Teacher**: Fast convergence, straight trajectories facilitate cleaner clustering in Subsystem 3 |
| **TreeFlow** ([Ramachandran & Sra, 2026](../01_ramachandran2026_trees_to_flows.md)) | Tree-partition conditioned flow field | Fast ($3 - 5$ steps, $2\times$ faster than diffusion) | Disentangles modes into independent tree partition sub-classes | **Adopted for Fine-Tuning**: Eliminates mode crossing along tree branches |

---

## 3. Interface with Downstream Subsystems

1. **To Subsystem 3 (Topological Induction):** Outputs the trajectory bundle $\Xi = \{\xi_i(\tau)\}_{i=1}^M$ whose phase-space geometry contains the bifurcations.
2. **To Subsystem 5 (DMP Action Leaves):** Provides the training data for fitting Dynamical Movement Primitive shape weights $\mathbf{w}_{\text{DMP}}$ and attractor velocities.

