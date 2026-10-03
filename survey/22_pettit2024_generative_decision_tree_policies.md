# Generative Design of Decision Tree Policies for Reinforcement Learning (DisCo-DSO)

- **Authors:** Jacob F. Pettit, Chak Shing Lee, Jiachen Yang, Alex Ho, Daniel Faissol, Brenden K. Petersen, Mikel Landajuela
- **Year / Venue:** 2024 / ICML 2024 Workshop on Structured Probabilistic Inference & Generative Modeling (SPIGM); extended as *DisCo-DSO: Coupling Discrete and Continuous Optimization for Efficient Generative Design in Hybrid Spaces* in AAAI 2025 / [arXiv:2412.11051](https://arxiv.org/abs/2412.11051)
- **Paper Link / Identifier:** [OpenReview: H6jtCfgX5N](https://openreview.net/pdf?id=H6jtCfgX5N) / [arXiv:2412.11051](https://arxiv.org/abs/2412.11051) / [DOI: 10.1609/aaai.v39i19.34407](https://doi.org/10.1609/aaai.v39i19.34407)
- **Primary Category:** Generative Policy Design & Hybrid Tree Structure Search (Pillar 3: Evolutionary & Hybrid Structure Learning / Policy Distillation)

---

### 1. Executive Summary & Core Hypothesis
Synthesizing interpretable, compact decision tree (DT) policies directly from reinforcement learning (RL) or black-box demonstrations is severely hampered by the **hybrid discrete-continuous nature** of tree structures: the discrete topology (which feature to branch on, tree depth, node connections) is tightly coupled with continuous parameters (split thresholds and leaf action constants). Traditional approaches either solve this sequentially (greedy top-down splitting like CART, followed by heuristic tuning) or use decoupled mixed-integer programming (MIP) or Bayesian optimization (BO), which scale poorly to deep trees and continuous control spaces.

The authors hypothesize that discrete-continuous tree generation can be formulated as an autoregressive probabilistic generative process. They introduce **DisCo-DSO (Discrete-Continuous Deep Symbolic Optimization)**, which represents a decision tree as an interleaved sequence of discrete structural tokens and continuous numeric tokens. An autoregressive recurrent neural network (RNN) samples the discrete skeleton, while conditioning continuous probability distributions (e.g., Gaussian mixtures or truncated normal distributions) to sample the corresponding split thresholds. The entire hybrid generator is trained end-to-end using risk-seeking policy gradients to directly maximize episode return or policy imitation fidelity.

---

### 2. Theoretical Framework & Mathematical Formulation

- **Interleaved Hybrid Token Representation:**
  A decision tree $\mathcal{T}$ of variable depth is serialized into a preorder traversal sequence $\mathbf{z} = (z_1, z_2, \dots, z_N)$. Unlike purely symbolic expressions, each split node $j$ requires a pair:
  $$z_{j,\text{disc}} \in \mathcal{V}_{\text{features}} \cup \mathcal{V}_{\text{ops}}, \quad z_{j,\text{cont}} \in \mathbb{R}$$
  where $z_{j,\text{disc}}$ selects the state feature $s_i$ and relational operator ($\le$), and $z_{j,\text{cont}}$ sets the numerical split threshold $\theta_j \in [\theta_{\min}, \theta_{\max}]$. Leaf nodes specify constant discrete actions $a \in \mathcal{A}$ or linear control weights.

- **Coupled Generative Factorization:**
  The joint probability of generating a complete hybrid tree $\mathcal{T} = (\mathbf{z}_{\text{disc}}, \mathbf{z}_{\text{cont}})$ decomposes autoregressively:
  $$p_\phi(\mathcal{T}) = \prod_{k=1}^K p_\phi(z_{k,\text{disc}} \mid z_{<k}) \cdot p_\phi(z_{k,\text{cont}} \mid z_{k,\text{disc}}, z_{<k})$$
  1. *Discrete Prior & Constraints:* $p_\phi(z_{k,\text{disc}} \mid z_{<k}) = \text{Softmax}(\mathbf{W}_d \mathbf{h}_k + \mathbf{m}_k)$, where $\mathbf{m}_k$ is a dynamic semantic mask enforcing valid tree structures (e.g., maximum depth limits, non-redundant parent-child feature splits).
  2. *Continuous Distribution:* $p_\phi(z_{k,\text{cont}} \mid \cdot) = \mathcal{N}\left(\mu_k(\mathbf{h}_k), \sigma_k^2(\mathbf{h}_k)\right)$, parameterized by dedicated output heads conditioned on the hidden state $\mathbf{h}_k$ of the discrete RNN.

- **Optimization Objective (Risk-Seeking Policy Gradient):**
  Rather than maximizing mean expected reward—which biases generation toward mediocre, safe trees—the framework optimizes the top $\epsilon$-quantile of generated designs (the *risk-seeking* objective):
  $$\mathcal{L}_{\text{risk}}(\phi) = -\mathbb{E}_{\mathcal{T} \sim p_\phi}\left[ \frac{\mathcal{R}(\mathcal{T}) - R_\epsilon(\phi)}{\sigma_R} \cdot \mathbb{I}(\mathcal{R}(\mathcal{T}) \ge R_\epsilon(\phi)) \nabla_\phi \log p_\phi(\mathcal{T}) \right]$$
  where $R_\epsilon(\phi)$ is the $(1-\epsilon)$-quantile of returns evaluated across a batch of candidate trees, concentrating gradient updates strictly on elite tree candidates.

---

### 3. Architecture & Neural Integration
- **Neural Role:** A multi-layer LSTM or GRU acts as the autoregressive meta-policy. It maintains an internal hidden state $\mathbf{h}_k$ tracking the hierarchical path from root to current node, allowing deep conditioning across nested tree levels.
- **Interface / Boundary:**
  - Evaluates black-box objective functions: a generated tree is compiled into an executable C++/Python policy, executed in the environment simulator, and its scalar episodic return $\mathcal{R}(\mathcal{T})$ is fed back as reinforcement signal.
  - Can alternatively be driven by behavioral cloning: $\mathcal{R}(\mathcal{T}) = -\text{MSE}(\pi_{\mathcal{T}}(s), \pi_{\text{teacher}}(s))$, acting as an interpretable policy distillation engine.

---

### 4. Empirical Evaluation & Benchmarks
- **Environments / Tasks:**
  - Classic Control & MuJoCo Gym benchmarks: CartPole-v1, MountainCar-v0, Pendulum-v1, LunarLander-v2.
  - High-dimensional symbolic regression and scientific discovery benchmarks.
- **Baseline Comparisons:**
  - Classical Tree Induction & Pruning: CART, VIPER (Verifiably Interpretable Policy Extraction via Reinforcement Learning).
  - Mixed Integer Optimization: MIO/MIP-based decision tree learners.
  - Evolutionary / Symbolic baselines: Standard DSO, Genetic Programming (gplearn).
- **Key Quantitative Results:**
  - Outperformed VIPER and standard CART distillation in sample efficiency and final policy reward while maintaining significantly shallower trees (depth $\le 4$).
  - DisCo-DSO found optimal non-linear split boundaries in hybrid spaces where decoupled continuous optimization failed to escape local minima.

---

### 5. Relevance & Critical Assessment for Flow2BT Distillation
- **Decision Trees vs. Behavior Trees Gap:**
  DisCo-DSO generates static **Decision Trees (DTs)** evaluated as single-pass state-action classifiers. It has **no native support for Behavior Tree (BT) mechanics**: no tick-based execution cycles, no tri-state returns (`SUCCESS`, `FAILURE`, `RUNNING`), no reactive fallback preemption (`?`), and no sequence memory (`->`).
- **Compatibility with Flow2BT Subsystems:**
  - *Incompatible with Subsystem 3 (Topological Bifurcation Induction):* Flow2BT extracts hierarchical tree structure analytically from continuous vector fields via reverse-time flow clustering. Using black-box combinatorial RL to search for tree topologies ignores the rich continuous velocity geometry $\mathbf{v}_\theta(\mathbf{x}_t, t)$.
  - *Highly Relevant for Subsystem 4 (Condition Node Induction):* In `src/flow2bt/conditions.py`, linear SVM hyperplanes currently suffer from feature collapse (dominated by agent speed, achieving only 66.8–72.3% routing fidelity, as documented in `IMPLEMENTATION_FINDINGS.md` §9.2). The autoregressive joint discrete-continuous tokenizer of DisCo-DSO can be adopted to synthesize multi-feature oblique condition predicates:
    $$C_k(\mathbf{s}) = \mathbb{I}\left(\sum_{j} w_j s_j + b \ge 0\right)$$
    over geometric features ($\text{TTC}, d_{\text{lat}}, v_{\text{rel}}$) to overcome greedy univariate split degradation.

