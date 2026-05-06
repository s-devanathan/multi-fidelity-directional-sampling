import numpy as np

from ads_multifidelity_budgeted import ADSMultiFidelityBudgeted, ADSConfig, Budget, ModelFidelity

def test_directional_estimator_returns_pf_and_ci():
    R, M = 5, 2

    def F1(u):
        u = np.asarray(u, float)
        return np.array([u[0] - 2.0, u[1]**2 + u[2] - 1.0], float)

    fidelities = {
        "F1": ModelFidelity("F1", F1, runtime_est=0.01),
        "F3": ModelFidelity("F3", F1, runtime_est=1.00),
    }

    cfg = ADSConfig(confidence=0.99, k_per_direction=2, boundary_scale=1.0)
    ads = ADSMultiFidelityBudgeted(R=R, M=M, fidelities=fidelities, base_name="F1", highest_name="F3", config=cfg)

    def mvm_stub(i, g_func):
        return 2.0, 0.5

    def mpp_stub(i, g_hi):
        u_star = np.zeros(R)
        u_star[0] = 2.0
        return u_star, float(np.linalg.norm(u_star))

    ads.initialize_mvm(mvm_stub)
    s = ads.run_iteration(N=2*R, budget=Budget(max_evals=50), eval_policy="highest", do_mpp=True, mpp_func=mpp_stub)

    assert "per_constraint" in s
    for _, ri in s["per_constraint"].items():
        assert "pf_mean" in ri
        assert "pf_ci" in ri