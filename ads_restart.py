from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, List
import os
import pickle
import numpy as np


@dataclass
class RestartPaths:
    prefix: str

    @property
    def pkl(self) -> str:
        return self.prefix + ".pkl"

    @property
    def npz(self) -> str:
        return self.prefix + ".npz"


def _ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def save_restart_state(
    paths: RestartPaths,
    *,
    rng: np.random.Generator,
    directions: List[np.ndarray],
    disc_X: List[np.ndarray],
    disc_Y: List[np.ndarray],
    disc_length_scale: Optional[float],
    cache: Dict[Tuple[str, Tuple[float, ...]], np.ndarray],
) -> None:
    _ensure_dir_for_file(paths.pkl)
    _ensure_dir_for_file(paths.npz)

    D = np.stack(directions, axis=0) if len(directions) else np.zeros((0, 0), dtype=float)
    X = np.stack(disc_X, axis=0) if len(disc_X) else np.zeros((0, 0), dtype=float)
    Y = np.stack(disc_Y, axis=0) if len(disc_Y) else np.zeros((0, 0), dtype=float)

    cache_items = list(cache.items())
    cache_keys = [k for (k, _) in cache_items]
    cache_vals = [v for (_, v) in cache_items]
    G = np.stack(cache_vals, axis=0) if len(cache_vals) else np.zeros((0, 0), dtype=float)

    np.savez_compressed(
        paths.npz,
        directions=D,
        disc_X=X,
        disc_Y=Y,
        cache_G=G,
    )

    meta: Dict[str, Any] = {
        "rng_bit_generator": rng.bit_generator.__class__.__name__,
        "rng_state": rng.bit_generator.state,
        "disc_length_scale": disc_length_scale,
        "cache_keys": cache_keys,
        "directions_shape": D.shape,
        "disc_X_shape": X.shape,
        "disc_Y_shape": Y.shape,
        "cache_G_shape": G.shape,
        "version": 1,
    }

    with open(paths.pkl, "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_restart_state(paths: RestartPaths) -> Dict[str, Any]:
    if not (os.path.exists(paths.pkl) and os.path.exists(paths.npz)):
        raise FileNotFoundError(f"Missing restart files: {paths.pkl} / {paths.npz}")

    with open(paths.pkl, "rb") as f:
        meta = pickle.load(f)

    data = np.load(paths.npz, allow_pickle=False)
    D = data["directions"]
    X = data["disc_X"]
    Y = data["disc_Y"]
    G = data["cache_G"]

    directions = [D[i].copy() for i in range(D.shape[0])] if D.size else []
    disc_X = [X[i].copy() for i in range(X.shape[0])] if X.size else []
    disc_Y = [Y[i].copy() for i in range(Y.shape[0])] if Y.size else []

    cache_keys = meta.get("cache_keys", [])
    if len(cache_keys) != (G.shape[0] if G.ndim == 2 else 0):
        raise ValueError("Restart cache key count does not match cache_G rows.")

    cache: Dict[Tuple[str, Tuple[float, ...]], np.ndarray] = {}
    for k, g in zip(cache_keys, (G if G.size else [])):
        cache[(k[0], tuple(k[1]))] = np.asarray(g, float).copy()

    return {
        "rng_state": meta["rng_state"],
        "rng_bit_generator": meta.get("rng_bit_generator"),
        "directions": directions,
        "disc_X": disc_X,
        "disc_Y": disc_Y,
        "disc_length_scale": meta.get("disc_length_scale", None),
        "cache": cache,
        "meta": meta,
    }