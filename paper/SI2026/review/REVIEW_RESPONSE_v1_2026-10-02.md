# Response to Reviewers: SI2026 Manuscript Review

**Document Version:** 1.0  
**Date:** 2026-10-02  
**Review Source:** `paper/AIS Lab. Manuscript Checker.html` (Session ID: `d053b517-0863-4010-8d4e-04f25bee76c9`, Review ID: `a849a11d-8258-4a8e-8e95-42d454084e7d`)  
**Manuscript:** `paper/SI2026/SICE-SI_en.tex`  
**Paper Title:** Distilled Behavior Tree-Based Modeling For Generative Crowd Behavior in Simulation Environment  
**Authors:** Nguyen Tri Tung Nguyen, Yuya Hosoda, and Joo-Ho Lee  

---

## 1. Executive Summary of Changes

In response to the automated and peer evaluation rubric from AIS Lab Manuscript Checker, we conducted a comprehensive revision of the manuscript (`SICE-SI_en.tex`). The key updates include:
1. **Simulation Protocol & Reproducibility:** Documented complete scenario generation parameters, agent count ($N=406$), empirical spawn timestamps, simulation horizon ($1,000$ frames at $5\,\text{fps}$), 5 randomized seeds with mean $\pm$ standard deviation, and formal definitions for sampling budgets ($1024 \times 16$).
2. **Reactivity & Interpretability Claims:** Replaced speculative environmental response assertions with precise, measured operational properties ($100\,\text{Hz}$ update capability / $10\,\text{ms}$ control cycles vs. $<0.05\,\text{ms}$ tree tick compute). Added explicit linear SVM hyperplane weights from the induced tree (`tree.txt`) demonstrating transparent rule routing.
3. **Calibrated Preservation Scope:** Explicitly separated macroscopic crowd realism (approximate preservation of Density and Population distributions) from microscopic trajectory accuracy, acknowledging and analyzing the performance trade-off in Kinematics, DTW path deviation, and Travel Time.
4. **Focused Safety/Kinematic Section:** Simplified the decentralized filtering discussion to realistic motor acceleration limits ($a_{\max} = 3.0\,\text{m/s}^2$) and semi-implicit Euler integration, removing speculative CBF collision invariance claims and the Waypoint + CBF baseline.
5. **Exact Teacher Architecture & Training Specification:** Added the complete 4-layer MLP configuration, CrowdES context encoder dimensions, AdamW optimizer parameters, and ETH training split details.
6. **Mathematical Rigor & Notation:** Formally defined the terminal displacement offset $\Delta\xi_{ij}(T_f)$, numerically stabilized the time-to-collision formula ($\epsilon = 10^{-4}\,\text{m/s}$), and corrected the abstract typo ("open-loop" $\to$ "closed-loop").
7. **Template Font Compliance:** Removed all custom `\fontsize` overrides, allowing typography to be natively determined by the official conference class (`sice-si.cls`).

---

## 2. Point-by-Point Response to Top Priorities

### Priority 1: Specify closed-loop simulation protocol and make trial comparisons comparable
- **Review Finding (`EXPERIMENT-CONDITIONS`, ID: `2c42d069-a1bd-455f-9c58-bfc577cbc44e`):**
  > *"The caption reports '20 for CrowdES, 5 for Flow2BT controllers,' but agent count, spawn/goal protocol, initial states, trial duration, random seeds, and the definition of '1024 × 16' are not given. Add a compact protocol paragraph or table stating the simulator/environment, number of agents, spawn-goal and initialization procedure, simulation horizon, random seeds, Flow2BT rollout/training sampling settings, and the meaning of '1024 × 16.' Report variability (e.g., mean ± standard deviation)."*
- **Revision Made:**
  - Added full protocol details to **Section 4.1 (Experimental Setup)**:
    - **Dataset & Agents:** Real-world ETH crowd benchmark (`seq_eth`, 406 total pedestrian agents).
    - **Spawn & Goal Mechanism:** Pedestrian agents spawn dynamically at recorded empirical timestamps and coordinates, tracking global NavMesh polylines toward exit boundaries.
    - **Simulation Horizon:** 1,000 frames ($200\,\text{s}$ at $5\,\text{fps}$, $0.2\,\text{s}$ simulator step).
    - **Random Seeds & Variability:** Evaluated over 5 randomized trials (seeds 0--4) with reported $\text{mean} \pm \text{std}$ across all metrics in Table 1.
    - **Definition of $1024 \times 16$:** Clarified in **Section 3.2** as $S = 1,024$ unique context states sampled across the scene with $R = 16$ stochastic rollouts per state ($M = 16,384$ total paths).

---

### Priority 2: Support claimed reactivity and interpretability with direct evidence
- **Review Finding (`EXPERIMENT-RESULTS`, ID: `b8c4bbfe-c60b-44f7-a814-336dae52c7d8`):**
  > *"The paper claims 'transparent, auditable decision logic' and 'response latency to 10 ms,' but reports only BT evaluation time and aggregate EMD metrics; no induced tree/rules or perturbation-based reaction example is shown. Distinguish the 10 ms control update period from measured computation time and limit the claim to a 100 Hz evaluation capability rather than demonstrated reactive response."*
- **Revision Made:**
  - In **Section 3.3 (Tree-to-Reactive BT Assembly)**, incorporated concrete hyperplane coefficients directly extracted from `tree.txt`:
    - **Root Guard 0 (Speed & Clearance, 92.7% accuracy):**
      $$-2.12 v_i - 0.358 d_{\text{nav}} + 0.091 v_{\text{close}} + 1.79 \ge 0$$
      separates slow/yielding agents from active forward locomotion.
    - **Subsequent Guards 4 & 6:** Evaluate heading deviation $\theta_{\text{nav}}$ (weights $+1.34$ and $+1.62$) and corridor clearance to branch into lateral swerve primitives (`swerve_left_3`, `swerve_right_6`) versus direct marching (`march_2`, `march_7`).
  - In **Section 1, Section 3.3, and Section 4.3**, decoupled terminology:
    - Update rate: $100\,\text{Hz}$ ($\Delta t = 10\,\text{ms}$ control period).
    - Ticking compute latency: measured at $<0.05\,\text{ms}$ per tick for evaluating the full tree hierarchy.
    - Formulated the benefit as a $100\,\text{Hz}$ reactive evaluation and preemption capability compared to the $2000\,\text{ms}$ open-loop chunking of prior generative models.

---

### Priority 3: Qualify preservation claims using observed performance gap
- **Review Finding (`EXPERIMENT-SCOPE-LIMIT`, ID: `9748f2c8-a639-4b85-b16c-c731fe2c5e06`, `81ce4dd4-7ab5-4975-b4db-86c964906a1d`):**
  > *"For Flow2BT (Flow), Kinem., DTW, and Time are '0.550, 2.19, 1.014,' versus CrowdES values '0.340, 1.62, 0.596,' while the text says it preserves realism 'without sacrificing' macro-level distributions. State explicitly that the current results support approximate preservation of the reported macroscopic density/population measures, but show degraded agent-level kinematic, path-alignment, and travel-time similarity relative to CrowdES."*
- **Revision Made:**
  - Removed all absolute claims ("without sacrificing realism").
  - Revised **Abstract, Section 1, Section 4.2, and Section 5** to explicitly state **"approximately preserves"** macroscopic Density and Population distributions.
  - Added dedicated analysis in **Section 4.2** addressing the microscopic gap:
    - Kinematics EMD ($0.550 \pm 0.041$ vs. $0.340$)
    - DTW path alignment ($2.19 \pm 0.064$ vs. $1.62$)
    - Travel Time EMD ($1.014 \pm 0.058$ vs. $0.596$)
    - Identified the structural origins of this trade-off: (1) discrete compression of continuous multimodal distributions into $B=8$ unimodal DMP attractors, and (2) finite-duration stop recovery where agents resuming from rest initiate slower acceleration profiles.

---

### Priority 4: Define safety-filter and simulation constraints needed to reproduce execution
- **Review Finding (`METHOD-SIMULATION`, ID: `4a09be82-7954-4a04-863c-b7b1015babca`, `60ee8648-f1b2-4a11-a5a3-5d719b7acb13`):**
  > *"The filter is described with '∥u_i∥∞ ≤ a_max' and 'h_ij(s_i,s_j) ≥ 0,' but neither a_max, h_ij, pedestrian radius/separation margin, nor solver/integration details are specified. Provide the operational definition of the barrier constraint, agent radius or minimum separation, acceleration bound, integration step, and how the filter is solved."*
- **Revision Made:**
  - Streamlined and focused this section under **Kinematic Acceleration Limits** (Section 3.3):
    - Defined physical acceleration limits: $\|\mathbf{u}_i\|_\infty \le a_{\max}$ with $a_{\max} = 3.0\,\text{m/s}^2$.
    - Integration schema: semi-implicit Euler integration executed at $\Delta t = 10\,\text{ms}$ ($100\,\text{Hz}$).
    - In accordance with author instructions, removed unneeded CBF forward-invariance claims, collision discussions, and the Waypoint + CBF baseline, keeping the paper focused cleanly on generative distillation and reactive BT execution.

---

## 3. Detailed Response to All Other Rubric Items

### `METHOD-LEARNING` (ID: `b5b5c1ea-b9e7-4372-8b3c-cad0953e8c97`)
- **Review Comment:** Missing teacher network architecture, optimizer, training split, and preprocessing.
- **Revision:**
  In **Section 3.1**, detailed:
  - Architecture: 4-layer MLP with hidden dimensions $[256, 512, 512, 256]$.
  - Temporal embeddings: 128-dimensional sinusoidal time embeddings.
  - Conditioning context: CrowdES context encoder with latent dimension 256 and hidden dimension 2048 over an 8-frame ($1.6\,\text{s}$) observation window.
  - Training configuration: AdamW optimizer, learning rate $10^{-4}$, Mean Squared Error flow matching loss $\mathcal{L}_{\text{CFM}}$, trained on the standard ETH training split.

### `COMMON-NOTATION` (ID: `94a48e76-879c-4519-9e0c-9b2b01a25daf`)
- **Review Comment:** Undefined $\Delta\xi_{ij}(T_f)$ in Eq. (5) and division-by-zero risk in TTC calculation.
- **Revision:**
  - Formally defined terminal displacement offset in **Section 3.2**:
    $$\Delta\xi_{ij}(T_f) \triangleq \xi_i(T_f) - \xi_j(T_f) \quad (\text{in meters})$$
  - Formally stabilized the TTC definition in **Section 3.1**:
    $$\text{TTC} = \frac{\|\mathbf{p}_j - \mathbf{p}_i\|}{\|\mathbf{v}_j - \mathbf{v}_i\| + \epsilon}$$
    with $\epsilon = 10^{-4}\,\text{m/s}$ for static or co-moving pedestrians.
  - Stated explicit physical units across all state and sensory quantities ($\text{m}$, $\text{m/s}$, $\text{m/s}^2$, $\text{s}$, $\text{rad}$).

### `INTRO-EXPERIMENT-FINDING` & `INTRO-NOVELTY-CONTRIBUTION` (ID: `93c7410a-1dcc-40e1-9634-48e9ce903c2a`)
- **Review Comment:** Calibrate introductory claims to match the empirical findings.
- **Revision:**
  Adjusted the contribution bullets in **Section 1** to reflect $100\,\text{Hz}$ update capability, approximate macroscopic preservation, and closed-form DMP execution while noting the microscopic trade-off.

### `EXPERIMENT-SCOPE-LIMIT` & `CONCLUSION-ACHIEVED-FUTURE` (ID: `76192d6c-c2d9-4177-9d3b-d9febec43426`, `5ce82365-3e84-4b08-b12e-70d55e5e69d0`)
- **Review Comment:** Explicitly bound claims to the tested benchmark and distinguish achievements from limitations.
- **Revision:**
  In **Section 4.1 and Section 5**, explicitly noted that evaluation is currently conducted on the ETH `seq_eth` benchmark, highlighted the microscopic fidelity gap as an inherent trade-off of discrete leaf clustering, and outlined future extensions to multimodal sensory inputs and dynamic 3D scenes.

### Abstract Consistency Check
- **Review Finding:** The original abstract inadvertently referenced *"open-loop robotic simulations"* while proposing a reactive framework intended for interactive closed-loop deployment.
- **Revision:** Corrected the text in `\abst{...}` to explicitly read *"closed-loop robotic simulations"*.

---

## 4. Typography and Template Conformance

- **Template:** `sice-si.cls` (SICE System Integration Division Annual Conference).
- **Font Sizing:** Reverted all custom `\fontsize{...}` commands in the bibliography and body text. The reference section is formatted by the native `thebibliography` environment of `sice-si.cls`, and Table 1 utilizes the standard `\footnotesize` sizing macro.
- **Compilation:** Clean compilation with `lualatex` and `bibtex` (`Exit code: 0`).
