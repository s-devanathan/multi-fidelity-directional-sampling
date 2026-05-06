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

def _key_u(u: np.ndarray, ndigits: int = 14) -> Tuple[float, ...]:
    return tuple(np.round(u.astype(float), ndigits))


# ----------------------------
# Direction management
# ----------------------------

@dataclass
class DirectionBank:
    R: int
    rng: np.random.Generator
    directions: List[np.ndarray] = field(default_factory=list)

    def sample_uniform(self, n: int) -> List[np.ndarray]:
        return [unit(self.rng.normal(size=self.R)) for _ in range(n)]

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
    err_estimator: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None

    def error_est(self, u: np.ndarray, g: np.ndarray) -> np.ndarray:
        if self.err_estimator is None:
            return np.zeros_like(g, dtype=float)
        return np.asarray(self.err_estimator(u, g), dtype=float)

@dataclass
class EvaluationCache:
    cache: Dict[Tuple[str, Tuple[float, ...]], np.ndarray] = field(default_factory=dict)

    def get(self, fidelity: str, u: np.ndarray) -> Optional[np.ndarray]:
        return self.cache.get((fidelity, _key_u(u)))

    def put(self, fidelity: str, u: np.ndarray, g: np.ndarray) -> None:
        self.cache[(fidelity, _key_u(u))] = np.asarray(g, dtype=float)


# ----------------------------
# Discrepancy model: per-constraint RBF
# ----------------------------

@dataclass
class RBFDiscrepancy:
    M: int
    eps: float = 1e-12
    length_scale: Optional[float] = None
    X: List[np.ndarray] = field(default_factory=list)
    Y: List[np.ndarray] = field(default_factory=list)

    def add(self, u: np.ndarray, delta: np.ndarray) -> None:
        self.X.append(np.asarray(u, float).copy())
        self.Y.append(np.asarray(delta, float).copy())

    def _infer_ls(self) -> float:
        if self.length_scale is not None:
            return float(self.length_scale)
        if len(self.X) < 2:
            return 1.0
        X = np.stack(self.X, axis=0)
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
        X = np.stack(self.X, axis=0)
        Y = np.stack(self.Y, axis=0)

        d2 = np.sum((X - u[None, :]) ** 2, axis=1)
        w = np.exp(-0.5 * d2 / (ls * ls))
        s = float(w.sum())
        if s < self.eps:
            return np.zeros(self.M, dtype=float)
        return (w[:, None] * Y).sum(axis=0) / s

    def uncertainty_proxy(self, u: np.ndarray) -> float:
        if len(self.X) == 0:
            return 1.0
        u = np.asarray(u, float)
        ls = self._infer_ls()
        X = np.stack(self.X, axis=0)
        d2 = np.sum((X - u[None, :]) ** 2, axis=1)
        w = np.exp(-0.5 * d2 / (ls * ls))
        wmax = float(np.max(w)) if w.size else 0.0
        return float(1.0 - wmax)


@dataclass
class CorrectedBaseModel:
    base: ModelFidelity
    discrepancy: RBFDiscrepancy

    def eval(self, u: np.ndarray) -> np.ndarray:
        return np.asarray(self.base.eval(u), float) + self.discrepancy.predict(u)


# ----------------------------
# Budgeting + candidate selection (Option B cost control)
# ----------------------------

@dataclass
class Budget:
    max_evals: Optional[int] = None
    max_time: Optional[float] = None  # seconds, using runtime_est

@dataclass
class Candidate:
    u: np.ndarray
    j: int                 # direction index
    i: int                 # constraint index that generated this band point
    sign: int              # -1 for t-, +1 for t+
    score: float = 0.0     # computed later

def min_abs_g(g: np.ndarray) -> float:
    return float(np.min(np.abs(g)))

def candidate_score(
    g_hat: np.ndarray,
    unc: float,
    boundary_scale: float,
    w_boundary: float,
    w_unc: float,
) -> float:
    close = math.exp(-min_abs_g(g_hat) / max(boundary_scale, 1e-12))
    return (w_boundary * close) + (w_unc * unc)


# ----------------------------
# Fidelity scheduler
# ----------------------------

@dataclass
class FidelityScheduler:
    fidelities: Dict[str, ModelFidelity]
    base_name: str
    highest_name: str
    cache: EvaluationCache
    corrected_model: CorrectedBaseModel

    boundary_scale: float = 1.0
    w_boundary: float = 1.0
    w_unc: float = 1.0
    runtime_w: float = 1.0

    def base(self) -> ModelFidelity:
        return self.fidelities[self.base_name]

    def highest(self) -> ModelFidelity:
        return self.fidelities[self.highest_name]

    def choose(self, u: np.ndarray) -> ModelFidelity:
        g_hat = self.corrected_model.eval(u)
        unc = self.corrected_model.discrepancy.uncertainty_proxy(u)
        base_score = candidate_score(g_hat, unc, self.boundary_scale, self.w_boundary, self.w_unc)

        best, best_val = None, -1e300
        for f in self.fidelities.values():
            val = base_score / (1.0 + self.runtime_w * f.runtime_est)
            if best is None or val > best_val or (abs(val - best_val) < 1e-12 and f.runtime_est > best.runtime_est):
                best, best_val = f, val
        return best

    def eval(self, f: ModelFidelity, u: np.ndarray) -> np.ndarray:
        cached = self.cache.get(f.name, u)
        if cached is not None:
            return cached
        g = f.eval(u)
        self.cache.put(f.name, u, g)
        return g


# ----------------------------
# Ray solver (corrected base)
# ----------------------------

@dataclass
class RayBoundarySolver:
    g_func: Callable[[np.ndarray], np.ndarray]
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
        flo = f_lo
        for _ in range(self.max_bisect):
            mid = 0.5 * (lo + hi)
            fmid = gi(mid)
            if abs(fmid) <= self.g_tol:
                return mid, "ok"
            if flo * fmid <= 0:
                hi = mid
            else:
                lo = mid
                flo = fmid
        return 0.5 * (lo + hi), "ok"


# ----------------------------
# Config + State
# ----------------------------

@dataclass
class ADSConfig:
    confidence: float = 0.99
    band_c0: float = 0.5

    novelty_dot_threshold: float = 0.92
    add_dirs_step: Optional[int] = None  # default R
    g_tol: float = 1e-6

    # selection controls for Option B:
    k_per_direction: int = 2
    k_per_constraint: int = 0

    # scoring weights:
    boundary_scale: float = 1.0
    w_boundary: float = 1.0
    w_unc: float = 1.0

@dataclass
class ADSState:
    beta: np.ndarray
    sigma: np.ndarray
    band_c: np.ndarray
    directions_used: int = 0


# ----------------------------
# ADS Option B engine with cost control
# ----------------------------

class ADSMultiFidelityBudgeted:
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
            boundary_scale=self.cfg.boundary_scale,
            w_boundary=self.cfg.w_boundary,
            w_unc=self.cfg.w_unc,
        )

        self.dirs = DirectionBank(R=R, rng=self.rng)

        self.ray = RayBoundarySolver(
            g_func=self.corrected.eval,
            M=M,
            g_tol=self.cfg.g_tol,
        )

        self.state = ADSState(
            beta=np.zeros(M, float),
            sigma=np.zeros(M, float),
            band_c=np.full(M, self.cfg.band_c0, float),
        )

    # Step 1 hook
    def initialize_mvm(self, mvm_func: Callable[[int, Callable[[np.ndarray], np.ndarray]], Tuple[float, float]]) -> None:
        for i in range(self.M):
            b, s = mvm_func(i, self.corrected.eval)
            self.state.beta[i] = float(b)
            self.state.sigma[i] = float(s)

    # Step 2: Option B candidates (2NM)
    def build_candidates(self, N: int) -> List[Candidate]:
        if N < 2 * self.R:
            raise ValueError("N must be >= 2R")
        self.dirs.ensure_n(N)

        cands: List[Candidate] = []
        for j in range(N):
            d = self.dirs.directions[j]
            for i in range(self.M):
                beta_i = float(self.state.beta[i])
                sigma_i = float(self.state.sigma[i])
                delta = float(self.state.band_c[i]) * sigma_i
                t_minus = max(1e-8, beta_i - delta)
                t_plus = max(t_minus * 1.0001, beta_i + delta)
                cands.append(Candidate(u=t_minus * d, j=j, i=i, sign=-1))
                cands.append(Candidate(u=t_plus * d, j=j, i=i, sign=+1))
        return cands

    def select_candidates_budgeted(self, cands: List[Candidate], budget: Budget) -> List[Candidate]:
        # score all
        for c in cands:
            g_hat = self.corrected.eval(c.u)
            unc = self.discrepancy.uncertainty_proxy(c.u)
            c.score = candidate_score(g_hat, unc, self.cfg.boundary_scale, self.cfg.w_boundary, self.cfg.w_unc)

        cands_sorted = sorted(cands, key=lambda x: x.score, reverse=True)

        selected: Dict[Tuple[float, ...], Candidate] = {}

        def add(c: Candidate) -> None:
            selected[_key_u(c.u)] = c

        # coverage per direction
        kdir = max(0, int(self.cfg.k_per_direction))
        if kdir > 0:
            by_j: Dict[int, List[Candidate]] = {}
            for c in cands_sorted:
                by_j.setdefault(c.j, []).append(c)
            for j, group in by_j.items():
                for c in group[:kdir]:
                    add(c)

        # optional per constraint
        kcon = max(0, int(self.cfg.k_per_constraint))
        if kcon > 0:
            by_i: Dict[int, List[Candidate]] = {}
            for c in cands_sorted:
                by_i.setdefault(c.i, []).append(c)
            for i, group in by_i.items():
                for c in group[:kcon]:
                    add(c)

        hi = self.scheduler.highest()

        def within_budget(n_evals: int, time_est: float) -> bool:
            if budget.max_evals is not None and n_evals > budget.max_evals:
                return False
            if budget.max_time is not None and time_est > budget.max_time:
                return False
            return True

        sel_list = list(selected.values())
        n_e = len(sel_list)
        t_e = n_e * hi.runtime_est

        for c in cands_sorted:
            if _key_u(c.u) in selected:
                continue
            if not within_budget(n_e + 1, t_e + hi.runtime_est):
                break
            add(c)
            n_e += 1
            t_e += hi.runtime_est

        return list(selected.values())

    # Step 3: evaluate selected candidates
    def evaluate_selected(
        self,
        selected: List[Candidate],
        policy: str = "highest",
    ) -> List[Tuple[str, np.ndarray, np.ndarray, Candidate]]:
        out = []
        for c in selected:
            if policy == "highest":
                f = self.scheduler.highest()
            elif policy == "scheduled":
                f = self.scheduler.choose(c.u)
            else:
                raise ValueError("policy must be 'highest' or 'scheduled'")
            g = self.scheduler.eval(f, c.u)
            out.append((f.name, c.u, np.asarray(g, float), c))
        return out

    # Step 4: update discrepancy
    def update_discrepancy(self, evals: List[Tuple[str, np.ndarray, np.ndarray, Candidate]]) -> None:
        base = self.scheduler.base()
        for fname, u, g, _ in evals:
            if fname == base.name:
                continue
            g0 = self.scheduler.eval(base, u)
            self.discrepancy.add(u, np.asarray(g, float) - np.asarray(g0, float))

    # Steps 5-7
    def directional_estimate(self, N: int) -> Dict[int, Dict[str, Any]]:
        rep: Dict[int, Dict[str, Any]] = {}
        self.state.directions_used = N

        for i in range(self.M):
            beta_i = float(self.state.beta[i])
            sigma_i = float(self.state.sigma[i])
            delta = float(self.state.band_c[i]) * sigma_i
            t_lo = max(1e-8, beta_i - delta)
            t_hi = max(t_lo * 1.0001, beta_i + delta)

            betas, pfs = [], []
            statuses = {"ok": 0, "always_safe": 0, "always_fail": 0, "unbracketed": 0}

            for j in range(N):
                d = self.dirs.directions[j]
                b, st = self.ray.find_beta(i=i, d=d, t_lo=t_lo, t_hi=t_hi)
                statuses[st] = statuses.get(st, 0) + 1
                if st == "ok" and b is not None:
                    betas.append(float(b))
                    pfs.append(pf_from_beta(float(b)))

            pf_mean, pf_lo, pf_hi = mean_and_ci(pfs, confidence=self.cfg.confidence)

            rep[i] = {
                "N_used": len(pfs),
                "statuses": statuses,
                "pf_mean": pf_mean,
                "pf_ci": (pf_lo, pf_hi),
                "beta_min": float(np.min(betas)) if betas else float("nan"),
                "beta_mean": float(np.mean(betas)) if betas else float("nan"),
            }

        return rep

    # Step 8 hook
    def mpp_search(self, i_star: int, mpp_func: Callable[[int, Callable[[np.ndarray], np.ndarray]], Tuple[np.ndarray, float]]) -> Tuple[np.ndarray, float]:
        hi = self.scheduler.highest()
        def g_hi(u: np.ndarray) -> np.ndarray:
            return self.scheduler.eval(hi, u)
        u_star, beta_star = mpp_func(i_star, g_hi)
        return np.asarray(u_star, float), float(beta_star)

    # Step 9 novelty
    def novelty_update(self, d_new: np.ndarray) -> bool:
        rho = self.dirs.most_similar_dot(d_new)
        if rho < self.cfg.novelty_dot_threshold:
            add_n = self.cfg.add_dirs_step or self.R
            self.dirs.add(self.dirs.sample_uniform(add_n))
            return True
        return False

    def run_iteration(
        self,
        N: int,
        budget: Optional[Budget] = None,
        eval_policy: str = "highest",
        do_mpp: bool = True,
        mpp_func: Optional[Callable[[int, Callable[[np.ndarray], np.ndarray]], Tuple[np.ndarray, float]]] = None,
        governing_selector: Optional[Callable[[Dict[int, Dict[str, Any]]], int]] = None,
    ) -> Dict[str, Any]:
        if N < 2 * self.R:
            raise ValueError("N must be >= 2R")

        budget = budget or Budget(max_evals=None, max_time=None)

        cands = self.build_candidates(N)
        selected = self.select_candidates_budgeted(cands, budget=budget)
        evals = self.evaluate_selected(selected, policy=eval_policy)
        self.update_discrepancy(evals)
        rep = self.directional_estimate(N)

        summary: Dict[str, Any] = {
            "N": N,
            "num_candidates_total": len(cands),
            "num_selected_evals": len(selected),
            "eval_policy": eval_policy,
            "eval_fidelity_counts": self._count_fidelities(evals),
            "discrepancy_data": len(self.discrepancy.X),
            "per_constraint": rep,
        }

        if do_mpp:
            if mpp_func is None:
                raise ValueError("do_mpp=True requires mpp_func.")
            if governing_selector is None:
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
    def _count_fidelities(evals: List[Tuple[str, np.ndarray, np.ndarray, Candidate]]) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for name, _, _, _ in evals:
            c[name] = c.get(name, 0) + 1
        return c

    # ---- Restart integration (requires ads_restart.py in repo)
    def save_restart(self, prefix: str) -> None:
        from ads_restart import RestartPaths, save_restart_state
        paths = RestartPaths(prefix=prefix)
        save_restart_state(
            paths,
            rng=self.rng,
            directions=self.dirs.directions,
            disc_X=self.discrepancy.X,
            disc_Y=self.discrepancy.Y,
            disc_length_scale=self.discrepancy.length_scale,
            cache=self.cache.cache,
        )

    def load_restart(self, prefix: str, *, strict: bool = True) -> None:
        from ads_restart import RestartPaths, load_restart_state
        st = load_restart_state(RestartPaths(prefix=prefix))

        if strict:
            bg_name = st.get("rng_bit_generator", None)
            if bg_name is not None and bg_name != self.rng.bit_generator.__class__.__name__:
                raise ValueError(f"Bit generator mismatch: saved={bg_name} current={self.rng.bit_generator.__class__.__name__}")

        self.rng.bit_generator.state = st["rng_state"]
        self.dirs.directions = [np.asarray(d, float) for d in st["directions"]]
        self.discrepancy.X = [np.asarray(x, float) for x in st["disc_X"]]
        self.discrepancy.Y = [np.asarray(y, float) for y in st["disc_Y"]]
        self.discrepancy.length_scale = st.get("disc_length_scale", None)

        if strict:
            unknown = set(k[0] for k in st["cache"].keys()) - set(self.scheduler.fidelities.keys())
            if unknown:
                raise ValueError(f"Restart cache contains unknown fidelities: {sorted(unknown)}")
            for (_, _), g in st["cache"].items():
                if g.shape != (self.M,):
                    raise ValueError(f"Cached g has shape {g.shape}, expected ({self.M},)")

        self.cache.cache = {(k[0], tuple(k[1])): np.asarray(v, float) for k, v in st["cache"].items()}