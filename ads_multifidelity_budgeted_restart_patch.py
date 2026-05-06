# Add these imports at top of your module:
# from ads_restart import RestartPaths, save_restart_state, load_restart_state

from typing import Optional

# Inside class ADSMultiFidelityBudgeted:

def save_restart(self, prefix: str) -> None:
    """
    Persist directions + discrepancy + cache + RNG state.
    """
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
    """
    Load a restart state into this ADS instance.

    strict=True performs basic dimension checks.
    Note: fidelity set must be compatible with cached fidelity names.
    """
    from ads_restart import RestartPaths, load_restart_state

    paths = RestartPaths(prefix=prefix)
    st = load_restart_state(paths)

    # Restore RNG state deterministically
    # (We assume same bit generator type; NumPy can still accept state dict even if class differs,
    #  but strict mode checks name for safety.)
    if strict:
        bg_name = st.get("rng_bit_generator", None)
        if bg_name is not None and bg_name != self.rng.bit_generator.__class__.__name__:
            raise ValueError(f"Bit generator mismatch: saved={bg_name} current={self.rng.bit_generator.__class__.__name__}")
    self.rng.bit_generator.state = st["rng_state"]

    # Restore directions
    self.dirs.directions = [np.asarray(d, float) for d in st["directions"]]

    # Restore discrepancy
    self.discrepancy.X = [np.asarray(x, float) for x in st["disc_X"]]
    self.discrepancy.Y = [np.asarray(y, float) for y in st["disc_Y"]]
    self.discrepancy.length_scale = st.get("disc_length_scale", None)

    # Restore cache
    # Optional strict check: fidelity names in cache exist in current fidelities
    if strict:
        unknown = set(k[0] for k in st["cache"].keys()) - set(self.scheduler.fidelities.keys())
        if unknown:
            raise ValueError(f"Restart cache contains unknown fidelities: {sorted(unknown)}")
        # also check g dimension matches M
        for (_, _), g in st["cache"].items():
            if g.shape != (self.M,):
                raise ValueError(f"Cached g has shape {g.shape}, expected ({self.M},)")

    self.cache.cache = {(k[0], tuple(k[1])): np.asarray(v, float) for k, v in st["cache"].items()}