import numpy as np

from ads_multifidelity_budgeted import ADSMultiFidelityBudgeted, ADSConfig, Budget, ModelFidelity

def test_budget_limits_number_of_evals():
    R, M = 4, 2

    def F1(u):
        u = np.asarray(u, float)
        return np.array([u[0] + u[1] - 2.0, u[2] + u[3] - 1.0], float)

    fidelities = {
        "F1": ModelFidelity("F1", F1, runtime_est=0.01),
        "F3": ModelFidelity("F3", F1, runtime_est=1.00),
    }

    cfg = ADSConfig(confidence=0.99, k_per_direction=2, boundary_scale=1.0)
    ads = ADSMultiFidelityBudgeted(R=R, M=M, fidelities=fidelities, base_name="F1", highest_name="F3", config=cfg)

    def mvm_stub(i, g_func):
        return 3.0, 0.4

    def mpp_stub(i, g_hi):
        u_star = np.zeros(R)
        u_star[0] = 3.0
        return u_star, float(np.linalg.norm(u_star))

    ads.initialize_mvm(mvm_stub)
    s = ads.run_iteration(N=2*R, budget=Budget(max_evals=10), eval_policy="highest", do_mpp=True, mpp_func=mpp_stub)

    assert s["num_selected_evals"] <= 10