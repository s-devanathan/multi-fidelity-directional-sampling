from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any
import math
import numpy as np


# ----------------------------
# Math / stats utilities
# ----------------------------

def unit(v: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Zero vector cannot be normalized.")
    return v / n

def phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def pf_from_beta(beta: float) -> float:
    return phi(-beta)

def z_value(confidence: float) -> float:
    # Common values. Extend as needed.
    if abs(confidence - 0.95) < 1e-12:
        return 1.959963984540054
    if abs(confidence - 0.99) < 1e-12:
        return 2.5758293035489004
    # Fallback: approximate inverse error function could be added.
    raise ValueError("Only 0.95 and 0.99 supported by default.")

def mean_and_ci(samples: Sequence[float], confidence: float) -> Tuple[float, float, float]:
    """
    Normal-approx CI on mean: mean ± z * s/sqrt(n)
    Returns (mean, lo, hi). If n<2, CI collapses to mean.
    """
    xs = np.asarray(samples, dtype=float)
    m = float(xs.mean()) if xs.size else float("nan")
    if xs.size < 2:
        return m, m, m
    s = float(xs.std(ddof=1))
    z = z_value(confidence)
    half = z * s / math.sqrt(xs.size)
    return m, m - half, m + half


# ----------------------------
# Direction sampling
# ----------------------------

@dataclass
class DirectionBank:
    R: int
    rng: np.random.Generator
    directions: List[np.ndarray] = field(default_factory=list)

    def sample_uniform(self, n: int) -> List[np.ndarray]:
        """Uniform random directions on the unit sphere."""
        ds = []
        for _ in range(n):
            v = self.rng.normal(size=self.R)
            ds.append(unit(v))
        return ds

    def add_directions(self, new_dirs: Sequence[np.ndarray]) -> None:
        for d in new_dirs:
            self.directions.append(unit(np.asarray(d, dtype=float)))

    def ensure_n(self, n_total: int) -> None:
        if len(self.directions) >= n_total:
            return
        self.add_directions(self.sample_uniform(n_total - len(self.directions)))

    def most_similar_dot(self, d: np.ndarray) -> float:
        """Returns max dot(d, existing). If none exist, returns -inf."""
        if not self.directions:
            return float("-inf")
        d = unit(d)
        dots = [float(np.dot(d, dj)) for dj in self.directions]
        return max(dots)

    def nearest_indices(self, d: np.ndarray, k: int) -> List[int]:
        d = unit(d)
        dots = np.array([np.dot(d, dj) for dj in self.directions], dtype=float)
        if dots.size == 0:
            return []
        idx = np.argsort(-dots)  # descending
        return idx[: min(k, idx.size)].tolist()


# ----------------------------
# Fidelity / model interfaces
# ----------------------------

@dataclass
class EvalResult:
    u: np.ndarray
    g: np.ndarray  # shape (M,)
    fidelity: str
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelFidelity:
    """
    A single fidelity model.

    - eval(u) -> g (M constraints)
    - runtime_est: estimated seconds per run
    - err_estimator: optional function (u, g)-> per-constraint error estimate array shape (M,)
      (Lower fidelity usually has larger error; HF may have near-zero error or known discretization error.)
    """
    name: str
    eval: Callable[[np.ndarray], np.ndarray]
    runtime_est: float
    err_estimator: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None

    def error(self, u: np.ndarray, g: np.ndarray) -> np.ndarray:
        if self.err_estimator is None:
            return np.zeros_like(g, dtype=float)
        return np.asarray(self.err_estimator(u, g), dtype=float)

class SurrogateUpdater:
    """
    Hook for multi-fidelity updating.
    You decide how LF models get updated when HF samples arrive.
    """
    def update(self, samples: List[EvalResult]) -> None:
        raise NotImplementedError

class NoOpUpdater(SurrogateUpdater):
    def update(self, samples: List[EvalResult]) -> None:
        return


# ----------------------------
# Multi-fidelity scheduling heuristic
# ----------------------------

@dataclass
class MultiFidelityManager:
    """
    Selects which fidelity to run for each candidate point.

    Simple default policy:
    - For band sampling (Step 3), run *highest* fidelity by default.
      (You can override to do staged fidelities if desired.)
    - For exploration / additional points, choose based on value score.
    """
    fidelities: Dict[str, ModelFidelity]

    def highest(self) -> ModelFidelity:
        # Highest fidelity is assumed to be the one with minimum error / max runtime.
        # If you have an explicit ordering, adapt this.
        return max(self.fidelities.values(), key=lambda f: f.runtime_est)

    def choose_for_point(
        self,
        u: np.ndarray,
        g_lf: Optional[np.ndarray] = None,
        lf_error_proxy: Optional[np.ndarray] = None,
        impact_scale: float = 1.0,
    ) -> ModelFidelity:
        """
        Cost-aware choice among fidelities.
        If g_lf is provided, prioritize higher fidelity near boundary / high error.
        """
        # If no context, pick the cheapest.
        if g_lf is None:
            return min(self.fidelities.values(), key=lambda f: f.runtime_est)

        # Compute impact proxy from LF: close to boundary => higher impact.
        # Using per-constraint min abs(g) as proxy.
        min_abs_g = float(np.min(np.abs(g_lf)))
        impact = math.exp(-min_abs_g / max(impact_scale, 1e-12))

        # Combine with error proxy if available.
        err_boost = 1.0
        if lf_error_proxy is not None:
            err_boost += float(np.max(lf_error_proxy))

        # Score = (impact * err_boost) / runtime
        best = None
        best_score = -1.0
        for f in self.fidelities.values():
            score = (impact * err_boost) / max(f.runtime_est, 1e-12)
            # Prefer higher fidelity if score ties (optional)
            if best is None or score > best_score:
                best = f
                best_score = score
        return best if best is not None else self.highest()


# ----------------------------
# 1D boundary solve along ray (LF)
# ----------------------------

@dataclass
class RayBoundarySolver:
    g_func: Callable[[np.ndarray], np.ndarray]  # returns (M,)
    M: int
    g_tol: float = 1e-6
    t_min: float = 1e-8
    max_expand: int = 8
    expand_factor: float = 1.6
    max_bisect: int = 60

    def find_beta_for_constraint(
        self,
        i: int,
        d: np.ndarray,
        t_lo: float,
        t_hi: float,
    ) -> Tuple[Optional[float], str]:
        """
        Find t where g_i(t d) = 0 via bracket+ bisection.
        Returns (beta, status) where status in {"ok","unbracketed","always_safe","always_fail"}.
        """
        d = unit(d)
        t_lo = max(self.t_min, float(t_lo))
        t_hi = max(t_lo * 1.0001, float(t_hi))

        def gi(t: float) -> float:
            u = t * d
            return float(self.g_func(u)[i])

        f_lo = gi(t_lo)
        f_hi = gi(t_hi)

        # If already bracketed:
        if f_lo == 0.0:
            return t_lo, "ok"
        if f_hi == 0.0:
            return t_hi, "ok"

        # Need opposite signs:
        expand_count = 0
        while f_lo * f_hi > 0 and expand_count < self.max_expand:
            # Expand interval outward (mostly on hi)
            t_hi = t_hi * self.expand_factor
            f_hi = gi(t_hi)
            expand_count += 1

        if f_lo * f_hi > 0:
            # Could be always safe or always fail along explored range
            if f_lo > 0 and f_hi > 0:
                return None, "always_safe"
            if f_lo < 0 and f_hi < 0:
                return None, "always_fail"
            return None, "unbracketed"

        # Bisection
        lo, hi = t_lo, t_hi
        flo, fhi = f_lo, f_hi
        for _ in range(self.max_bisect):
            mid = 0.5 * (lo + hi)
            fmid = gi(mid)
            if abs(fmid) <= self.g_tol:
                return mid, "ok"
            # keep bracket
            if flo * fmid <= 0:
                hi, fhi = mid, fmid
            else:
                lo, flo = mid, fmid
        return 0.5 * (lo + hi), "ok"


# ----------------------------
# MVM placeholders (you plug real ones)
# ----------------------------

@dataclass
class MVMEstimate:
    beta: float
    sigma: float

def mean_value_method_stub(i: int) -> MVMEstimate:
    """
    Placeholder. Replace with your MVM calculation on (low-fidelity) model.
    Should return beta_i and sigma_i.
    """
    # Dummy defaults
    return MVMEstimate(beta=3.0, sigma=0.4)


# ----------------------------
# ADS Orchestrator
# ----------------------------

@dataclass
class ADSConfig:
    N0: int
    confidence: float = 0.99
    band_c0: float = 0.5
    band_c_min: float = 0.25
    band_c_max: float = 2.0
    novelty_dot_threshold: float = 0.92
    add_dirs_step: Optional[int] = None  # if None, use R
    g_tol: float = 1e-6

@dataclass
class ADSState:
    beta_mvm: np.ndarray          # (M,)
    sigma_mvm: np.ndarray         # (M,)
    band_c: np.ndarray            # (M,)
    # per-constraint results storage
    betas_dir: Dict[int, List[float]] = field(default_factory=dict)
    pf_dir: Dict[int, List[float]] = field(default_factory=dict)
    curvature_dir: Dict[int, List[float]] = field(default_factory=dict)
    # caches
    samples: List[EvalResult] = field(default_factory=list)

class ADSRunner:
    def __init__(
        self,
        R: int,
        M: int,
        lf_model: ModelFidelity,
        mf_manager: MultiFidelityManager,
        updater: SurrogateUpdater,
        config: ADSConfig,
        rng: Optional[np.random.Generator] = None,
    ):
        self.R = R
        self.M = M
        self.lf_model = lf_model
        self.mf = mf_manager
        self.updater = updater
        self.cfg = config
        self.rng = rng or np.random.default_rng()
        self.dirs = DirectionBank(R=R, rng=self.rng)

        self.state = ADSState(
            beta_mvm=np.zeros(M, dtype=float),
            sigma_mvm=np.zeros(M, dtype=float),
            band_c=np.full(M, config.band_c0, dtype=float),
        )

        # LF boundary solver uses LF model callable
        self.ray_solver = RayBoundarySolver(
            g_func=self.lf_model.eval,
            M=M,
            g_tol=self.cfg.g_tol,
        )

    # ---- Step 1
    def initialize_with_mvm(self, mvm_func: Callable[[int], MVMEstimate] = mean_value_method_stub) -> None:
        for i in range(self.M):
            est = mvm_func(i)
            self.state.beta_mvm[i] = est.beta
            self.state.sigma_mvm[i] = est.sigma

    # ---- Step 2
    def make_band_points(self, N: int) -> Dict[int, List[Tuple[int, float, np.ndarray]]]:
        """
        Returns dict i -> list of (j, t, u) band points for each constraint.
        For each (i,j): two points at t- and t+.
        """
        if N < 2 * self.R:
            raise ValueError(f"N must be >= 2R. Got N={N}, R={self.R}.")

        self.dirs.ensure_n(N)
        pts: Dict[int, List[Tuple[int, float, np.ndarray]]] = {i: [] for i in range(self.M)}

        for i in range(self.M):
            beta_i = float(self.state.beta_mvm[i])
            sigma_i = float(self.state.sigma_mvm[i])
            c_i = float(self.state.band_c[i])
            delta = c_i * sigma_i

            t_minus = max(1e-8, beta_i - delta)
            t_plus = max(t_minus * 1.0001, beta_i + delta)

            for j, d in enumerate(self.dirs.directions[:N]):
                u_minus = t_minus * d
                u_plus = t_plus * d
                pts[i].append((j, t_minus, u_minus))
                pts[i].append((j, t_plus, u_plus))
        return pts

    # ---- Step 3 (supports >2 fidelities)
    def evaluate_band_points(
        self,
        band_points: Dict[int, List[Tuple[int, float, np.ndarray]]],
        fidelity_policy: str = "highest",
    ) -> List[EvalResult]:
        """
        Evaluate chosen fidelity at band points.
        Default: highest fidelity for all band points.
        """
        results: List[EvalResult] = []
        f_hi = self.mf.highest()

        for i in range(self.M):
            for (j, t, u) in band_points[i]:
                if fidelity_policy == "highest":
                    f = f_hi
                else:
                    # Example: choose based on LF prediction at u
                    g_lf = self.lf_model.eval(u)
                    err_lf = self.lf_model.error(u, g_lf)
                    f = self.mf.choose_for_point(u, g_lf=g_lf, lf_error_proxy=err_lf)

                g = f.eval(u)
                results.append(EvalResult(u=u, g=np.asarray(g, dtype=float), fidelity=f.name, meta={"i": i, "j": j, "t": t}))
        self.state.samples.extend(results)
        return results

    # ---- Step 4
    def update_low_fidelity(self, new_samples: List[EvalResult]) -> None:
        self.updater.update(new_samples)

    # ---- Step 5-7
    def directional_pf_estimate(self, N: int) -> Dict[int, Dict[str, Any]]:
        """
        For each constraint i:
          - solve beta_{i,j} on LF along each direction
          - compute pf_{i,j}
          - aggregate mean + CI
        Returns report per constraint.
        """
        report: Dict[int, Dict[str, Any]] = {}
        self.state.betas_dir.clear()
        self.state.pf_dir.clear()

        for i in range(self.M):
            beta_i = float(self.state.beta_mvm[i])
            sigma_i = float(self.state.sigma_mvm[i])
            delta = float(self.state.band_c[i]) * sigma_i
            t_lo = max(1e-8, beta_i - delta)
            t_hi = max(t_lo * 1.0001, beta_i + delta)

            betas_ij: List[float] = []
            pfs_ij: List[float] = []
            statuses: Dict[str, int] = {"ok": 0, "always_safe": 0, "always_fail": 0, "unbracketed": 0}

            for j, d in enumerate(self.dirs.directions[:N]):
                beta_ij, status = self.ray_solver.find_beta_for_constraint(i=i, d=d, t_lo=t_lo, t_hi=t_hi)
                statuses[status] = statuses.get(status, 0) + 1
                if status == "ok" and beta_ij is not None:
                    betas_ij.append(float(beta_ij))
                    pfs_ij.append(pf_from_beta(float(beta_ij)))

            self.state.betas_dir[i] = betas_ij
            self.state.pf_dir[i] = pfs_ij

            pf_mean, pf_lo, pf_hi = mean_and_ci(pfs_ij, confidence=self.cfg.confidence) if pfs_ij else (float("nan"), float("nan"), float("nan"))

            report[i] = {
                "N_used": len(pfs_ij),
                "statuses": statuses,
                "pf_mean": pf_mean,
                "pf_ci": (pf_lo, pf_hi),
                "beta_dir_mean": float(np.mean(betas_ij)) if betas_ij else float("nan"),
                "beta_dir_min": float(np.min(betas_ij)) if betas_ij else float("nan"),
            }

        return report

    # ---- Step 8: HF MPP search hook (user supplies)
    def mpp_search_high_fidelity(
        self,
        i_star: int,
        mpp_func: Callable[[int, ModelFidelity], Tuple[np.ndarray, float]],
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        mpp_func should return (u_star, beta_star). Uses highest fidelity model handle.
        """
        f_hi = self.mf.highest()
        u_star, beta_star = mpp_func(i_star, f_hi)
        d_new = unit(u_star)
        return u_star, float(beta_star), d_new

    # ---- Step 9: novelty check and direction enrichment
    def maybe_add_directions_from_new_dir(self, d_new: np.ndarray) -> bool:
        """
        Returns True if directions were added (and thus caller should restart from Step 2).
        """
        rho_max = self.dirs.most_similar_dot(d_new)
        if rho_max < self.cfg.novelty_dot_threshold:
            add_n = self.cfg.add_dirs_step or self.R
            self.dirs.add_directions(self.dirs.sample_uniform(add_n))
            return True
        return False

    # ---- Steps 10-11: acceptability test hook
    def lf_error_acceptability(
        self,
        pf_old: float,
        pf_new: float,
        ci_old: Tuple[float, float],
    ) -> bool:
        """
        Accept if change is within old CI half width (99% by default).
        """
        lo, hi = ci_old
        if not (math.isfinite(pf_old) and math.isfinite(pf_new) and math.isfinite(lo) and math.isfinite(hi)):
            return False
        half = 0.5 * (hi - lo)
        return abs(pf_new - pf_old) <= half

    # ---- Step 12: curvature/SORM placeholder
    def compute_curvature_placeholder(self, i: int, beta_ij: float, d: np.ndarray) -> float:
        """
        Placeholder: user should implement curvature estimation (from LF geometry, HF checks, etc.)
        """
        return 0.0

    # ---- Step 12-13: compute SORM directional pf using stored curvature (placeholder)
    def sorm_pf_placeholder(self, beta: float, K: float) -> float:
        """
        Placeholder for SORM formula of choice.
        For now, returns FORM pf.
        """
        return pf_from_beta(beta)

    # ---- Full iteration controller (one pass)
    def run_once(
        self,
        N: int,
        mpp_constraint_selector: Optional[Callable[[Dict[int, Dict[str, Any]]], int]] = None,
        mpp_func: Optional[Callable[[int, ModelFidelity], Tuple[np.ndarray, float]]] = None,
        do_mpp: bool = True,
        band_fidelity_policy: str = "highest",
    ) -> Dict[str, Any]:
        """
        Runs Steps 2-7 (+ optionally 8-13) once, given MVM already initialized.

        Returns a summary dict with per-constraint pf and CIs, plus bookkeeping.
        """
        # Step 2
        band_pts = self.make_band_points(N)

        # Step 3
        new_hf = self.evaluate_band_points(band_pts, fidelity_policy=band_fidelity_policy)

        # Step 4
        self.update_low_fidelity(new_hf)

        # Steps 5-7
        report = self.directional_pf_estimate(N)

        summary = {
            "N": N,
            "per_constraint": report,
            "num_new_samples": len(new_hf),
            "total_samples_cached": len(self.state.samples),
        }

        # Step 8 onward (optional)
        if do_mpp:
            if mpp_func is None:
                raise ValueError("do_mpp=True requires mpp_func.")
            if mpp_constraint_selector is None:
                # default: smallest beta_dir_min among constraints
                def mpp_constraint_selector(rep: Dict[int, Dict[str, Any]]) -> int:
                    best_i, best_beta = None, float("inf")
                    for i, ri in rep.items():
                        bmin = ri.get("beta_dir_min", float("inf"))
                        if math.isfinite(bmin) and bmin < best_beta:
                            best_beta = bmin
                            best_i = i
                    if best_i is None:
                        return 0
                    return best_i

            i_star = mpp_constraint_selector(report)
            u_star, beta_star, d_new = self.mpp_search_high_fidelity(i_star, mpp_func)

            added = self.maybe_add_directions_from_new_dir(d_new)
            summary["mpp"] = {"i_star": i_star, "u_star": u_star, "beta_star": beta_star, "direction_new": d_new, "added_directions": added}

            if added:
                # Caller should re-run from Step 2 with larger N (or same N but new dirs exist)
                return summary

            # Steps 12-13 (placeholder curvature + SORM update)
            for i in range(self.M):
                betas_ij = self.state.betas_dir.get(i, [])
                Ks = []
                pfs_sorm = []
                for j, beta_ij in enumerate(betas_ij):
                    d = self.dirs.directions[j]
                    K = self.compute_curvature_placeholder(i, beta_ij, d)
                    Ks.append(K)
                    pfs_sorm.append(self.sorm_pf_placeholder(beta_ij, K))
                self.state.curvature_dir[i] = Ks
                pf_mean, pf_lo, pf_hi = mean_and_ci(pfs_sorm, confidence=self.cfg.confidence) if pfs_sorm else (float("nan"), float("nan"), float("nan"))
                summary["per_constraint"][i]["pf_sorm_mean"] = pf_mean
                summary["per_constraint"][i]["pf_sorm_ci"] = (pf_lo, pf_hi)

        return summary