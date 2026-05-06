import numpy as np

from ads_multifidelity_budgeted import (
    ADSMultiFidelityBudgeted,
    ADSConfig,
    Budget,
    ModelFidelity,
)

R, M = 6, 3

def F1(u):
    u = np.asarray(u, float)
    return np.array([
        u[0] + 0.2*u[1] - 2.0,
        u[2]**2 + u[3] - 1.0,
        u[4] + u[5] - 1.5
    ], float)

def F2(u):
    return F1(u) + 0.05*np.sin(np.sum(u))

def F3(u):
    return F2(u) + 0.01*np.cos(np.sum(u[:2]))

fidelities = {
    "F1": ModelFidelity("F1", F1, runtime_est=0.01),
    "F2": ModelFidelity("F2", F2, runtime_est=0.10),
    "F3": ModelFidelity("F3", F3, runtime_est=1.00),
}

cfg = ADSConfig(
    confidence=0.99,
    k_per_direction=2,
    boundary_scale=0.5,
)

ads = ADSMultiFidelityBudgeted(
    R=R, M=M,
    fidelities=fidelities,
    base_name="F1",
    highest_name="F3",
    config=cfg
)

def mvm_stub(i, g_func):
    return 3.0, 0.4

ads.initialize_mvm(mvm_stub)

def mpp_stub(i, g_hi):
    u_star = np.zeros(R)
    u_star[0] = 3.0
    beta = float(np.linalg.norm(u_star))
    return u_star, beta

summary = ads.run_iteration(
    N=4*R,
    budget=Budget(max_evals=200),
    eval_policy="scheduled",
    do_mpp=True,
    mpp_func=mpp_stub,
)

print("Fidelity counts:", summary["eval_fidelity_counts"])
for i, ri in summary["per_constraint"].items():
    print(i, "pf", ri["pf_mean"], "CI", ri["pf_ci"], "beta_min", ri["beta_min"])