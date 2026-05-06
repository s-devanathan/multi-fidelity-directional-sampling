import numpy as np
import os
import tempfile

from ads_multifidelity_budgeted import (
    ADSMultiFidelityBudgeted,
    ADSConfig,
    Budget,
    ModelFidelity,
)

R, M = 4, 2

def F1(u):
    u = np.asarray(u, float)
    return np.array([u[0] + u[1] - 2.0,
                     u[2]**2 + u[3] - 1.0], float)

def F2(u):
    return F1(u) + 0.05*np.sin(np.sum(u))

def F3(u):
    return F2(u) + 0.01*np.cos(np.sum(u[:2]))

fidelities = {
    "F1": ModelFidelity("F1", F1, runtime_est=0.01),
    "F2": ModelFidelity("F2", F2, runtime_est=0.10),
    "F3": ModelFidelity("F3", F3, runtime_est=1.00),
}

cfg = ADSConfig(confidence=0.99, k_per_direction=2, boundary_scale=0.5)

def mvm_stub(i, g_func):
    return 3.0, 0.3

def mpp_stub(i, g_hi):
    u_star = np.zeros(R)
    u_star[0] = 2.5
    return u_star, float(np.linalg.norm(u_star))

with tempfile.TemporaryDirectory() as td:
    prefix = os.path.join(td, "checkpoint_ads")

    ads1 = ADSMultiFidelityBudgeted(R=R, M=M, fidelities=fidelities, base_name="F1", highest_name="F3", config=cfg)
    ads1.initialize_mvm(mvm_stub)
    ads1.run_iteration(N=2*R, budget=Budget(max_evals=50), eval_policy="scheduled", do_mpp=True, mpp_func=mpp_stub)
    ads1.save_restart(prefix)

    ads2 = ADSMultiFidelityBudgeted(R=R, M=M, fidelities=fidelities, base_name="F1", highest_name="F3", config=cfg)
    ads2.initialize_mvm(mvm_stub)
    ads2.load_restart(prefix)

    s2 = ads2.run_iteration(N=3*R, budget=Budget(max_evals=80), eval_policy="scheduled", do_mpp=True, mpp_func=mpp_stub)
    print("Resume OK. Discrepancy data:", s2["discrepancy_data"])
    print("Fidelity counts:", s2["eval_fidelity_counts"])