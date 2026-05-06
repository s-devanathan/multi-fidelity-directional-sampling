from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any
import math
import numpy as np


# ----------------------------
# Basic math / stats
# ----------------------------

def unit(v: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector.")
    return v / n

def phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def pf_from_beta(beta: float) -> float:
    return phi(-beta)

def z_value(confidence: float) -> float:
    if abs(confidence - 0.95) < 1e-12:
        return 1.959963984540054
    if abs(confidence - 0.99) < 1e-12:
        return 2.5758293035489004
    raise ValueError("Supported: 0.95, 0.99")

def mean_and_ci(samples: List[float], confidence: float) -> Tuple[float, float, float]:
    if len(samples) == 0:
        return float("nan"), float("nan"), float("nan")
    xs = np.asarray(samples, dtype=float)
    m = float(xs.mean())
    if xs.size < 2:
        return m, m, m
    s = float(xs.std(ddof=1))
    half = z_value(confidence) * s / math.sqrt(xs.size)
    return m, m - half, m + half


# ----------------------------
# Direction management
# ----------------------------

@dataclass
class DirectionBank:
    R: int
    rng: np.random.Generator
    directions: List[np.ndarray] = field(default_factory=list)

    def sample_uniform(self, n: int) -> List[np.ndarray]:
        ds = []
        for _ in range(n):
            ds.append(unit(self.rng.normal(size=self.R)))
        return ds

    def ensure_n(self, n_total: int) -> None:
        if len(self.directions) < n_total:
            self.directions.extend(self.sample_uniform(n_total - len(self.directions)))

    def add(self, dirs: List[np.ndarray]) -> None:
        self.directions.extend([unit(np.asarray(d, float)) for d in dirs])

    def most_similar_dot(self, d: np.ndarray) -> float:
        if not self.directions:
            return float("-inf")
        d = unit(d)
        return max(float(np.dot(d, dj)) for dj in self.directions)


# ----------------------------
# Fidelity models + caching
# ----------------------------

@dataclass
class ModelFidelity:
    name: str
    eval: Callable[[np.ndarray], np.ndarray]  # returns g shape (M,)
    runtime_est: float
    # Optional: per-constraint error estimate produced by the solver itself
    err_estimator: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None

    def error_est(self, u: np.ndarray, g: np.ndarray) -> np.ndarray:
        if self.err_estimator is None:
            return np.zeros_like(g, dtype=float)
        return np.asarray(self.err_estimator(u, g), dtype=float)

def _key_u(u: np.ndarray, ndigits: int = 14) -> Tuple[float, ...]:
    # stable cache key; adjust ndigits if needed
    return tuple(np.round(u.astype(float), ndigits))

@dataclass
class EvaluationCache:
    # cache[(fidelity_name, u_key)] = g
    cache: Dict[Tuple[str, Tuple[float, ...]], np.ndarray] = field(default_factory=dict)

    def get(self, fidelity: str, u: np.ndarray) -> Optional[np.ndarray]:
        return self.cache.get((fidelity, _key_u(u)))

    def put(self, fidelity: str, u: np.ndarray, g: np.ndarray) -> None:
        self.cache[(fidelity, _key_u(u))] = np.asarray(g, dtype=float)


# ----------------------------
# Discrepancy model: per-constraint RBF (simple, dependency-free)
# ----------------------------

@dataclass
class RBFDiscrepancy:
    """
    Learns delta(u) = g_hi(u) - g_base(u), vector-valued length M.

    Uses isotropic Gaussian RBF weights with a distance-based scale.
    This is NOT a GP; it's a smooth interpolant/regressor.
    """
    M: int
    eps: float = 1e-12
    length_scale: Optional[float] = None  # if None, inferred from data
    X: List[np.ndarray] = field(default_factory=list)   # u points
    Y: List[np.ndarray] = field(default_factory=list)   # delta vectors

    def add(self, u: np.ndarray, delta: np.ndarray) -> None:
        self.X.append(np.asarray(u, float).copy())
        self.Y.append(np.asarray(delta, float).copy())

    def _infer_ls(self) -> float:
        if self.length_scale is not None:
            return float(self.length_scale)
        if len(self.X) < 2:
            return 1.0
        X = np.stack(self.X, axis=0)
        # median pairwise distance heuristic (subsample if large)
        n = X.shape[0]
        idx = np.random.choice(n, size=min(n, 50), replace=False)
        Xs = X[idx]
        dists = []
        for i in range(Xs.shape[0]):
            for j in range(i + 1, Xs.shape[0]):
                dists.append(np.linalg.norm(Xs[i] - Xs[j]))
        med = np.median(dists) if dists else 1.0
        return float(max(med, 1e-3))

    def predict(self, u: np.ndarray) -> np.ndarray:
        if len(self.X) == 0:
            return np.zeros(self.M, dtype=float)
        u = np.asarray(u, float)
        ls = self._infer_ls()
        X = np.stack(self.X, axis=0)   # (n,R)
        Y = np.stack(self.Y, axis=0)   # (n,M)

        # Gaussian kernel weights
        d2 = np.sum((X - u[None, :]) ** 2, axis=1)
        w = np.exp(-0.5 * d2 / (ls * ls))
        s = float(w.sum())
        if s < self.eps:
            return np.zeros(self.M, dtype=float)
        return (w[:, None] * Y).sum(axis=0) / s

    def uncertainty_proxy(self, u: np.ndarray) -> float:
        """
        Returns a heuristic uncertainty proxy in [0,1] where 1 means "far from data".
        Based on max kernel weight at u.
        """
        if len(self.X) == 0:
            return 1.0
        u = np.asarray(u, float)
        ls = self._infer_ls()
        X = np.stack(self.X, axis=0)
        d2 = np.sum((X - u[None, :]) ** 2, axis=1)
        w = np.exp(-0.5 * d2 / (ls * ls))
        wmax = float(np.max(w)) if w.size else 0.0
        return float(1.0 - wmax)


# ----------------------------
# Corrected cheap model used for ray tracing
# ----------------------------

@dataclass
class CorrectedBaseModel:
    base: ModelFidelity
    discrepancy: RBFDiscrepancy

    def eval(self, u: np.ndarray) -> np.ndarray:
        g0 = self.base.eval(u)
        d = self.discrepancy.predict(u)
        return np.asarray(g0, float) + np.asarray(d, float)


# ----------------------------
# Fidelity scheduler (decides which simulator to run at a point)
# ----------------------------

@dataclass
class FidelityScheduler:
    fidelities: Dict[str, ModelFidelity]  # e.g. {"F1":..., "F2":..., "F3":...}
    base_name: str                        # cheapest solver name (for corrected LF)
    highest_name: str                     # highest fidelity solver name
    cache: EvaluationCache
    corrected_model: CorrectedBaseModel

    # Heuristic parameters
    boundary_scale: float = 1.0           # typical |g| scale; tune
    unc_w: float = 1.0                    # weight of discrepancy uncertainty
    boundary_w: float = 1.0               # weight of boundary closeness
    runtime_w: float = 1.0                # weight of runtime penalty

    def highest(self) -> ModelFidelity:
        return self.fidelities[self.highest_name]

    def base(self) -> ModelFidelity:
        return self.fidelities[self.base_name]

    def choose(self, u: np.ndarray) -> ModelFidelity:
        """
        Choose fidelity based on:
          - predicted closeness to boundary (from corrected base model)
          - discrepancy uncertainty proxy (distance to discrepancy data)
          - runtime
        Returns the fidelity maximizing value score.
        """
        u = np.asarray(u, float)
        g_hat = self.corrected_model.eval(u)
        closeness = math.exp(-float(np.min(np.abs(g_hat))) / max(self.boundary_scale, 1e-12))
        unc = float(self.corrected_model.discrepancy.uncertainty_proxy(u))

        best_f = None
        best_score = -1e300
        for f in self.fidelities.values():
            # Score increases with closeness+uncertainty, decreases with runtime
            score_num = (self.boundary_w * closeness) + (self.unc_w * unc)
            score = score_num / (1.0 + self.runtime_w * f.runtime_est)

            # tiny tie-break toward higher fidelity
            if best_f is None or score > best_score or (abs(score - best_score) < 1e-12 and f.runtime_est > best_f.runtime_est):
                best_f, best_score = f, score
        return best_f

    def eval(self, fidelity: ModelFidelity, u: np.ndarray) -> np.ndarray:
        cached = self.cache.get(fidelity.name, u)
        if cached is not None:
            return cached
        g = fidelity.eval(u)
        self.cache.put(fidelity.name, u, g)
        return g


# ----------------------------
# Ray boundary solver on corrected base model
# ----------------------------

@dataclass
class RayBoundarySolver:
    g_func: Callable[[np.ndarray], np.ndarray]  # corrected LF returning (M,)
    M: int
    g_tol: float = 1e-6
    t_min: float = 1e-8
    max_expand: int = 8
    expand_factor: float = 1.6
    max_bisect: int = 60

    def find_beta(self, i: int, d: np.ndarray, t_lo: float, t_hi: float) -> Tuple[Optional[float], str]:
        d = unit(d)
        t_lo = max(self.t_min, float(t_lo))
        t_hi = max(t_lo * 1.0001, float(t_hi))

        def gi(t: float) -> float:
            return float(self.g_func(t * d)[i])

        f_lo = gi(t_lo)
        f_hi = gi(t_hi)

        if f_lo == 0.0:
            return t_lo, "ok"
        if f_hi == 0.0:
            return t_hi, "ok"

        expand = 0
        while f_lo * f_hi > 0.0 and expand < self.max_expand:
            t_hi *= self.expand_factor
            f_hi = gi(t_hi)
            expand += 1

        if f_lo * f_hi > 0.0:
            if f_lo > 0 and f_hi > 0:
                return None, "always_safe"
            if f_lo < 0 and f_hi < 0:
                return None, "always_fail"
            return None, "unbracketed"

        lo, hi = t_lo, t_hi
        flo, fhi = f_lo, f_hi
        for _ in range(self.max_bisect):
            mid = 0.5 * (lo + hi)
            fmid = gi(mid)
            if abs(fmid) <= self.g_tol:
                return mid, "ok"
            if flo * fmid <= 0:
                hi, fhi = mid, fmid
            else:
                lo, flo = mid, fmid
        return 0.5 * (lo + hi), "ok"


# ----------------------------
# Configuration + State
# ----------------------------

@dataclass
class ADSConfig:
    confidence: float = 0.99
    band_c0: float = 0.5
    band_c_min: float = 0.25
    band_c_max: float = 2.0

    novelty_dot_threshold: float = 0.92
    add_dirs_step: Optional[int] = None  # default R

    g_tol: float = 1e-6

    # band sampling fidelity policy:
    # "highest": always highest
    # "scheduled": scheduler chooses per point
    band_policy: str = "highest"

@dataclass
class ADSState:
    beta: np.ndarray      # (M,)
    sigma: np.ndarray     # (M,)
    band_c: np.ndarray    # (M,)

    directions_used: int = 0

    # Directional results per constraint
    beta_ij: Dict[int, List[float]] = field(default_factory=dict)
    pf_ij: Dict[int, List[float]] = field(default_factory=dict)
    status_counts: Dict[int, Dict[str, int]] = field(default_factory=dict)


# ----------------------------
# Main ADS engine
# ----------------------------

class ADSMultiFidelity:
    def __init__(
        self,
        R: int,
        M: int,
        fidelities: Dict[str, ModelFidelity],
        base_name: str,
        highest_name: str,
        rng: Optional[np.random.Generator] = None,
        config: Optional[ADSConfig] = None,
    ):
        self.R = R
        self.M = M
        self.cfg = config or ADSConfig()
        self.rng = rng or np.random.default_rng()

        if base_name not in fidelities:
            raise ValueError("base_name not found in fidelities.")
        if highest_name not in fidelities:
            raise ValueError("highest_name not found in fidelities.")

        self.cache = EvaluationCache()
        self.discrepancy = RBFDiscrepancy(M=M)
        self.corrected = CorrectedBaseModel(base=fidelities[base_name], discrepancy=self.discrepancy)

        self.scheduler = FidelityScheduler(
            fidelities=fidelities,
            base_name=base_name,
            highest_name=highest_name,
            cache=self.cache,
            corrected_model=self.corrected,
        )

        self.dirs = DirectionBank(R=R, rng=self.rng)

        self.ray = RayBoundarySolver(
            g_func=self.corrected.eval,
            M=M,
            g_tol=self.cfg.g_tol,
        )

        # initialize beta/sigma later via MVM hook
        self.state = ADSState(
            beta=np.zeros(M, float),
            sigma=np.zeros(M, float),
            band_c=np.full(M, self.cfg.band_c0, float),
        )

    # ---- Step 1 hook
    def initialize_mvm(self, mvm_func: Callable[[int, Callable[[np.ndarray], np.ndarray]], Tuple[float, float]]) -> None:
        """
        mvm_func(i, g_func) -> (beta_i, sigma_i) using *some* model.
        Usually use base solver or corrected solver if you already have data.
        """
        for i in range(self.M):
            beta_i, sigma_i = mvm_func(i, self.corrected.eval)
            self.state.beta[i] = float(beta_i)
            self.state.sigma[i] = float(sigma_i)

    # ---- Step 2
    def band_points(self, N: int) -> List[np.ndarray]:
        """
        Because one run returns all M constraints, we create 2N points total
        (using *a governing* band based on min beta/sigma or per-constraint?).

        To stay faithful to your per-constraint banding, we take a conservative union:
        for each direction we generate points for each constraint band and then deduplicate.
        That can exceed 2N but still far less than 2NM in many cases.

        If you want exactly 2N points, use a single band from i* (governing constraint).
        """
        if N < 2 * self.R:
            raise ValueError(f"N must be >= 2R. Got N={N}, R={self.R}.")

        self.dirs.ensure_n(N)
        pts = []

        for j in range(N):
            d = self.dirs.directions[j]
            # build per-constraint t- and t+ and add them all
            for i in range(self.M):
                beta_i = float(self.state.beta[i])
                sigma_i = float(self.state.sigma[i])
                delta = float(self.state.band_c[i]) * sigma_i
                t_minus = max(1e-8, beta_i - delta)
                t_plus = max(t_minus * 1.0001, beta_i + delta)
                pts.append(t_minus * d)
                pts.append(t_plus * d)

        # Deduplicate by rounding key
        uniq: Dict[Tuple[float, ...], np.ndarray] = {}
        for u in pts:
            uniq[_key_u(u)] = u
        return list(uniq.values())

    # ---- Step 3 (multi-fidelity)
    def evaluate_points(self, points: List[np.ndarray], policy: Optional[str] = None) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """
        Returns list of (fidelity_name, u, g).
        """
        policy = policy or self.cfg.band_policy
        out = []
        for u in points:
            u = np.asarray(u, float)
            if policy == "highest":
                f = self.scheduler.highest()
            elif policy == "scheduled":
                f = self.scheduler.choose(u)
            else:
                raise ValueError("policy must be 'highest' or 'scheduled'.")

            g = self.scheduler.eval(f, u)
            out.append((f.name, u, np.asarray(g, float)))
        return out

    # ---- Step 4 (update discrepancy using any higher-than-base evals)
    def update_discrepancy(self, evals: List[Tuple[str, np.ndarray, np.ndarray]]) -> None:
        base = self.scheduler.base()
        for fname, u, g in evals:
            if fname == base.name:
                continue
            g0 = self.scheduler.eval(base, u)  # cached or computed
            delta = np.asarray(g, float) - np.asarray(g0, float)
            self.discrepancy.add(u, delta)

    # ---- Steps 5-7
    def directional_estimate(self, N: int) -> Dict[int, Dict[str, Any]]:
        self.state.beta_ij.clear()
        self.state.pf_ij.clear()
        self.state.status_counts.clear()

        report: Dict[int, Dict[str, Any]] = {}

        for i in range(self.M):
            beta_i = float(self.state.beta[i])
            sigma_i = float(self.state.sigma[i])
            delta = float(self.state.band_c[i]) * sigma_i
            t_lo = max(1e-8, beta_i - delta)
            t_hi = max(t_lo * 1.0001, beta_i + delta)

            betas = []
            pfs = []
            statuses = {"ok": 0, "always_safe": 0, "always_fail": 0, "unbracketed": 0}

            for j in range(N):
                d = self.dirs.directions[j]
                b, st = self.ray.find_beta(i=i, d=d, t_lo=t_lo, t_hi=t_hi)
                statuses[st] = statuses.get(st, 0) + 1
                if st == "ok" and b is not None:
                    betas.append(float(b))
                    pfs.append(pf_from_beta(float(b)))

            self.state.beta_ij[i] = betas
            self.state.pf_ij[i] = pfs
            self.state.status_counts[i] = statuses

            pf_mean, pf_lo, pf_hi = mean_and_ci(pfs, confidence=self.cfg.confidence)

            report[i] = {
                "N_used": len(pfs),
                "statuses": statuses,
                "pf_mean": pf_mean,
                "pf_ci": (pf_lo, pf_hi),
                "beta_min": float(np.min(betas)) if betas else float("nan"),
                "beta_mean": float(np.mean(betas)) if betas else float("nan"),
            }

        self.state.directions_used = N
        return report

    # ---- Step 8 hook: HF MPP search (user provided)
    def mpp_search(self, i_star: int, mpp_func: Callable[[int, Callable[[np.ndarray], np.ndarray]], Tuple[np.ndarray, float]]) -> Tuple[np.ndarray, float]:
        """
        mpp_func(i, g_hi_eval) -> (u_star, beta_star)
        where g_hi_eval(u) returns (M,) from highest fidelity simulator.
        """
        hi = self.scheduler.highest()
        def g_hi(u: np.ndarray) -> np.ndarray:
            return self.scheduler.eval(hi, u)
        u_star, beta_star = mpp_func(i_star, g_hi)
        return np.asarray(u_star, float), float(beta_star)

    # ---- Step 9
    def novelty_update(self, d_new: np.ndarray) -> bool:
        rho_max = self.dirs.most_similar_dot(d_new)
        if rho_max < self.cfg.novelty_dot_threshold:
            add_n = self.cfg.add_dirs_step or self.R
            self.dirs.add(self.dirs.sample_uniform(add_n))
            return True
        return False

    # ---- One ADS iteration (2-7 + optional 8-9)
    def run_iteration(
        self,
        N: int,
        do_mpp: bool = True,
        mpp_func: Optional[Callable[[int, Callable[[np.ndarray], np.ndarray]], Tuple[np.ndarray, float]]] = None,
        governing_selector: Optional[Callable[[Dict[int, Dict[str, Any]]], int]] = None,
        band_policy: Optional[str] = None,
    ) -> Dict[str, Any]:
        if N < 2 * self.R:
            raise ValueError("N must be >= 2R")

        # Step 2
        points = self.band_points(N)

        # Step 3
        evals = self.evaluate_points(points, policy=band_policy)

        # Step 4
        self.update_discrepancy(evals)

        # Steps 5-7
        rep = self.directional_estimate(N)

        summary: Dict[str, Any] = {
            "N": N,
            "num_points_evaluated": len(points),
            "eval_fidelity_counts": self._count_fidelities(evals),
            "per_constraint": rep,
            "discrepancy_data": len(self.discrepancy.X),
        }

        # Step 8-9 (optional)
        if do_mpp:
            if mpp_func is None:
                raise ValueError("do_mpp=True requires mpp_func.")
            if governing_selector is None:
                # default: smallest beta_min
                def governing_selector(rep: Dict[int, Dict[str, Any]]) -> int:
                    best_i, best_b = 0, float("inf")
                    for i, ri in rep.items():
                        b = ri.get("beta_min", float("inf"))
                        if math.isfinite(b) and b < best_b:
                            best_b, best_i = b, i
                    return best_i

            i_star = governing_selector(rep)
            u_star, beta_star = self.mpp_search(i_star, mpp_func)
            d_new = unit(u_star)

            added = self.novelty_update(d_new)

            summary["mpp"] = {
                "i_star": i_star,
                "u_star": u_star,
                "beta_star": beta_star,
                "direction_new": d_new,
                "added_directions": added,
            }

        return summary

    @staticmethod
    def _count_fidelities(evals: List[Tuple[str, np.ndarray, np.ndarray]]) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for name, _, _ in evals:
            c[name] = c.get(name, 0) + 1
        return c