# Adaptive Directional Sampling (ADS) Multi‑Fidelity Reliability Meta‑Algorithm (U‑Space)

**Current date:** 2026-05-06  
**Scope:** Heuristics-based reliability estimation in **U-space** (standard normal space after Nataf transformation), using **Adaptive Directional Sampling (ADS)** with **discrete simulator fidelity levels** (e.g. F1/F2/F3).  
**Assumption:** One simulator run returns **all** `M` constraint values `g(u) ∈ ℝ^M`.

This package provides:
- A budgeted ADS loop with **Option B union of constraint bands** (up to `2*N*M` candidates).
- Multi-fidelity evaluation policies (`highest` or `scheduled`).
- A corrected base model `g_corr = g_base + δ̂` for fast directional root finding.
- Deterministic **restart / resume**: directions, discrepancy data, evaluation cache, RNG state.

---

## 1. Problem setup (U‑space)

Let:

- `u ∈ ℝ^R` be the transformed random vector in U-space, `u ~ N(0, I)`.
- There are `M` constraints / limit-state functions:  
  \[
  g_i(u) \le 0 \quad \text{is failure for constraint } i.
  \]
- We sample `N` unit directions `{d_j}` on the unit sphere:
  \[
  \|d_j\| = 1, \quad j=1,\dots,N
  \]
  and search along rays:
  \[
  u(t)= t d_j,\quad t \ge 0.
  \]
- For each constraint `i` and direction `j`, define the boundary distance (reliability index along that direction):
  \[
  \beta_{i,j} = \text{the } t \text{ such that } g_i(t d_j)=0.
  \]
- Convert to a directional probability of failure sample:
  \[
  p_{f,i,j} = \Phi(-\beta_{i,j}).
  \]
- Aggregate estimate:
  \[
  \hat p_{f,i} = \frac{1}{N_i} \sum_{j\in \mathcal{J}_i} p_{f,i,j}
  \]
  where `𝒥_i` is the set of directions for which we successfully found a boundary crossing.

---

## 2. Multi‑fidelity setting (discrete simulators)

We have discrete simulator levels:
- F1: cheapest, highest error
- F2: intermediate
- F3: highest fidelity (ground truth), most expensive

Each fidelity returns `g(u)` of length `M` in one call.

### 2.1 Corrected base model (for fast ray searches)

Directional boundary searches are done on a cheap model, improved using higher-fidelity samples.

Let the base solver be `F_base` (typically F1). Define:
\[
g_{corr}(u) = g_{base}(u) + \hat\delta(u),
\]
where discrepancy:
\[
\delta(u) = g_{hi}(u) - g_{base}(u)
\]
is learned online from higher-fidelity evaluations.

**Implementation choice here:** a simple **Gaussian RBF regression** discrepancy model trained on points `(u_k, δ(u_k))`. Swap for GP/co-kriging if needed.

---

## 3. Algorithm outline (ADS, Option B union + budgeted selection)

### Step 1: MVM initialization
For each constraint `i`, estimate:
- `β_i` reliability index,
- `σ_i` spread proxy (from MVM).

This implementation expects you to provide a callable:
`mvm_func(i, g_func) -> (beta_i, sigma_i)`.

### Step 2: Directions
Require `N ≥ 2R` (recommend `N ≈ 4R` initially). Sample unit directions uniformly.

### Step 2 (Option B): Union candidates for all constraints
For each constraint `i` define band half-width:
\[
\Delta_i = c_i \sigma_i
\]
(default `c_i=0.5`) and create two points per direction:
\[
t^-_i = \max(\epsilon, \beta_i - \Delta_i),\quad t^+_i = \beta_i + \Delta_i.
\]
Candidates:
\[
u^-_{i,j} = t^-_i d_j, \quad u^+_{i,j} = t^+_i d_j.
\]
Up to `2*N*M` candidates.

### Cost control (budgeted down-selection)
To keep Option B affordable:
- Score candidates by boundary proximity (under `g_corr`) and discrepancy uncertainty (distance to discrepancy data).
- Enforce coverage constraints:
  - at least `k_per_direction` candidates per direction
  - optionally at least `k_per_constraint` per constraint
- Then fill remaining budget by score.

Budget is:
- `Budget(max_evals=..., max_time=...)`.

### Step 3: Evaluate selected points (multi-fidelity)
Two policies:
- `eval_policy="highest"`: always F3
- `eval_policy="scheduled"`: choose among F1/F2/F3 per point based on value score / runtime

### Step 4: Update discrepancy
For any evaluation at fidelity != base:
- compute `δ(u)=g_f(u)-g_base(u)`
- add `(u, δ(u))` to discrepancy training data.

### Step 5: Directional boundary search on corrected base model
For each constraint i and direction j:
- bracket expansion + bisection on `g_corr,i(t d_j)=0`
- compute `β_{i,j}` if found

### Step 6-7: pf estimate + CI
Compute `p_{f,i,j}=Φ(-β_{i,j})`, then mean and CI (95% or 99%).

### Step 8-9: MPP (hook) + novelty direction check
You provide:
`mpp_func(i_star, g_hi) -> (u_star, beta_star)` on the highest fidelity callable.

If MPP direction is novel (dot product < threshold), add more directions and repeat.

---

## 4. Restart / resume (deterministic)

Restart state stores:
- RNG state (`numpy` bit generator state)
- directions sampled so far
- discrepancy training points
- evaluation cache (for all fidelities)

Files:
- `<prefix>.pkl` metadata, RNG state, cache keys
- `<prefix>.npz` numeric arrays (directions, discrepancy X/Y, cache values)

Resume:
- instantiate a new ADS object with the same fidelities
- call `load_restart(prefix)`
- continue from Step 2 with additional `N` or budget

---

## 5. Running examples

```bash
python examples/example_basic_run.py
python examples/example_restart_resume.py
```

---

## 6. Running tests

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

---

## 7. Next extensions (not included)
- Implement HLRF/FORM internally (finite difference gradients) instead of an external hook.
- Add curvature sampling near the MPP and SORM corrections.
- Replace RBF discrepancy with GP/co-kriging for better uncertainty quantification.