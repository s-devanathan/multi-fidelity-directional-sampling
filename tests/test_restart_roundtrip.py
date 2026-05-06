import numpy as np
import os
import tempfile

from ads_multifidelity_budgeted import ADSMultiFidelityBudgeted, ADSConfig, Budget, ModelFidelity
from ads_restart import RestartPaths, load_restart_state

def test_restart_roundtrip_creates_files_and_loads():
    R, M = 3, 2

    def F1(u):
        u = np.asarray(u, float)
        return np.array([u[0] - 1.0, u[1] + u[2] - 0.5], float)

    def F2(u):
        return F1(u) + 0.1*np.sin(np.sum(u))

    def F3(u):
        return F2(u) + 0.01*np.cos(np.sum(u))

    fidelities = {
        "F1": ModelFidelity("F1", F1, runtime_est=0.01),
        "F2": ModelFidelity("F2", F2, runtime_est=0.10),
        "F3": ModelFidelity("F3", F3, runtime_est=1.00),
    }

    cfg = ADSConfig(confidence=0.99, k_per_direction=2, boundary_scale=1.0)

    def mvm_stub(i, g_func):
        return 2.0, 0.5

    def mpp_stub(i, g_hi):
        u_star = np.zeros(R)
        u_star[0] = 2.0
        return u_star, float(np.linalg.norm(u_star))

    with tempfile.TemporaryDirectory() as td:
        prefix = os.path.join(td, "ads_state")

        ads = ADSMultiFidelityBudgeted(R=R, M=M, fidelities=fidelities, base_name="F1", highest_name="F3", config=cfg)
        ads.initialize_mvm(mvm_stub)
        ads.run_iteration(N=2*R, budget=Budget(max_evals=30), eval_policy="scheduled", do_mpp=True, mpp_func=mpp_stub)

        ads.save_restart(prefix)
        assert os.path.exists(prefix + ".pkl")
        assert os.path.exists(prefix + ".npz")

        st = load_restart_state(RestartPaths(prefix))
        assert "rng_state" in st
        assert isinstance(st["directions"], list)

        ads2 = ADSMultiFidelityBudgeted(R=R, M=M, fidelities=fidelities, base_name="F1", highest_name="F3", config=cfg)
        ads2.initialize_mvm(mvm_stub)
        ads2.load_restart(prefix)

        s2 = ads2.run_iteration(N=3*R, budget=Budget(max_evals=40), eval_policy="scheduled", do_mpp=True, mpp_func=mpp_stub)
        assert "per_constraint" in s2