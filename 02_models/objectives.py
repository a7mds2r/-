"""
objectives.py
=============
Robust Multi-Footprint Environmental Optimization for AI Systems
under Compound Uncertainty (EQCAM dataset).

Decision vector x (5-D, all normalized in [0,1] unless noted):
    x0 : cooling_setpoint_scale      -> mapped to [0.90, 1.10]  (multiplier on CoolingPower)
    x1 : battery_dispatch_threshold  -> mapped to [0.20, 0.80]  (SOC threshold to prefer battery)
    x2 : workload_shift_fraction     -> mapped to [0.00, 0.30]  (fraction of IT_Load shiftable to
                                                                   high-renewable periods)
    x3 : renewable_priority_weight   -> mapped to [0.00, 1.00]  (weight favoring PV/Wind over grid)
    x4 : curtailment_tolerance       -> mapped to [0.00, 1.00]  (willingness to accept curtailment
                                                                   to protect battery SOH)

Uncertainty vector xi = a resampled scenario batch drawn from the empirical EQCAM
time series (weather, load, risk regime). Compound uncertainty = joint draw across
weather (GHI/Wind/Temp), workload (CPU/GPU/Memory util), and grid-risk regime (RAC).

Three composite objectives (per user decision: merge 5 raw footprints -> 3 objectives):
    f1 = Energy Footprint      (minimize)  -- effective grid-drawn energy incl. cooling
    f2 = Carbon/Deficit Footprint (minimize) -- based on Net Energy Balance (NEB) deficit
    f3 = Reliability/Degradation Footprint (minimize) -- battery health + risk-tier penalty

All objectives are robustified via a convex combination of the mean and the
worst-case (max) over B bootstrap scenario batches drawn under compound uncertainty:
    F_robust = (1 - beta) * mean_b(F_b) + beta * max_b(F_b),   beta in (0,1]
"""

import numpy as np
import pandas as pd

BOUNDS = np.array([
    [0.90, 1.10],   # x0 cooling_setpoint_scale
    [0.20, 0.80],   # x1 battery_dispatch_threshold
    [0.00, 0.30],   # x2 workload_shift_fraction
    [0.00, 1.00],   # x3 renewable_priority_weight
    [0.00, 1.00],   # x4 curtailment_tolerance
])
DIM = BOUNDS.shape[0]
N_OBJ = 3
ROBUST_BETA = 0.35   # weight on worst-case scenario in robust aggregation
N_SCENARIO_BATCHES = 5
SCENARIO_BATCH_SIZE = 256


def decode(x_unit: np.ndarray) -> np.ndarray:
    """Map a unit-cube candidate x in [0,1]^5 to physical bounds."""
    x_unit = np.clip(x_unit, 0.0, 1.0)
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    return lo + x_unit * (hi - lo)


class EQCAMScenarioEngine:
    """Loads EQCAM data and draws compound-uncertainty scenario batches."""

    def __init__(self, csv_path: str, seed: int = 42):
        df = pd.read_csv(csv_path)
        df["RAC_num"] = df["RAC"].map({"Low": 0.0, "Medium": 0.5, "High": 1.0})
        self.df = df
        self.n = len(df)
        self.rng_master = np.random.default_rng(seed)
        # Precompute risk-tier groups once (stratified compound-uncertainty sampling)
        self._groups = {k: v.index.values for k, v in df.groupby("RAC")}
        # Precompute numpy column arrays once (avoids repeated .values / DataFrame slicing)
        cols = ["PV_AC", "WindPower", "IT_Load", "CoolingPower", "Battery_SOC", "Battery_SOH",
                "ChargePower", "DischargePower", "StorageEff", "RAC_num", "CurtailmentFlag"]
        self._arrays = {c: df[c].to_numpy() for c in cols}
        self._batch_cache = {}

    def draw_batches(self, n_batches: int, batch_size: int, run_seed: int):
        """Return n_batches sets of row-index arrays (stratified compound-uncertainty draws).
        Cached per (n_batches, batch_size, run_seed) so a whole generation's evaluate_population
        call reuses the SAME scenario draws across all candidates (fair, and much faster)."""
        key = (n_batches, batch_size, run_seed)
        if key in self._batch_cache:
            return self._batch_cache[key]
        rng = np.random.default_rng(run_seed)
        batches = []
        for _ in range(n_batches):
            idx_parts = []
            for tier, idxs in self._groups.items():
                take = max(1, int(batch_size * (len(idxs) / self.n)))
                idx_parts.append(rng.choice(idxs, size=take, replace=True))
            idx = np.concatenate(idx_parts)
            rng.shuffle(idx)
            idx = idx[:batch_size]
            batches.append(idx)
        self._batch_cache[key] = batches
        return batches

    def batch_arrays(self, idx):
        """Return dict of numpy arrays for the given row indices (no DataFrame overhead)."""
        return {c: arr[idx] for c, arr in self._arrays.items()}


def _simulate_batch(x_phys: np.ndarray, batch: dict) -> np.ndarray:
    """
    Simulate the 3 raw footprints for one scenario batch given physical decision x.
    `batch` is a dict of numpy arrays (see EQCAMScenarioEngine.batch_arrays).
    Returns array [f1_energy, f2_carbon, f3_reliability] (all >=0, minimize).
    """
    cooling_scale, batt_thresh, shift_frac, ren_weight, curtail_tol = x_phys

    pv = batch["PV_AC"]
    wind = batch["WindPower"]
    it_load = batch["IT_Load"]
    cooling = batch["CoolingPower"]
    soc = batch["Battery_SOC"] / 100.0
    soh = batch["Battery_SOH"] / 100.0
    charge = batch["ChargePower"]
    discharge = batch["DischargePower"]
    storage_eff = np.clip(batch["StorageEff"] / 100.0, 0.5, 0.999)
    nrisk = batch["RAC_num"]
    curtail_flag = batch["CurtailmentFlag"]

    # --- Effective controllable load after workload shifting ---
    renewable_avail = pv + wind
    shiftable = it_load * shift_frac
    # more shifting effective when renewables are abundant relative to load
    ren_ratio = np.clip(renewable_avail / (it_load + 1e-6), 0, 3)
    shift_efficacy = np.clip(ren_ratio / 3.0, 0, 1)
    effective_load = it_load - shiftable * shift_efficacy

    effective_cooling = cooling * cooling_scale

    # --- Battery dispatch policy ---
    battery_supports = (soc > batt_thresh).astype(float)
    battery_supply = battery_supports * np.minimum(discharge, effective_load) * storage_eff

    grid_and_renewable_need = np.maximum(effective_load + effective_cooling - battery_supply, 0.0)
    renewable_used = np.minimum(grid_and_renewable_need, renewable_avail * (0.5 + 0.5 * ren_weight))
    grid_draw = np.maximum(grid_and_renewable_need - renewable_used, 0.0)

    # f1: Energy footprint -> normalized total effective + grid-drawn energy
    f1 = grid_draw + 0.15 * effective_cooling

    # f2: Carbon/deficit footprint -> penalize negative NEB analog (grid deficit),
    #     curtailment tolerance trades off unused renewables against battery protection
    deficit = grid_draw - curtail_tol * curtail_flag * renewable_avail * 0.05
    f2 = np.maximum(deficit, 0.0) * (1.0 + 0.5 * nrisk)

    # f3: Reliability/degradation footprint -> battery stress + risk-tier penalty
    charge_stress = np.abs(charge - discharge) / (np.abs(charge) + np.abs(discharge) + 1e-6)
    soh_penalty = (1.0 - soh) * 100.0
    f3 = soh_penalty + 20.0 * charge_stress * (1 - battery_supports) + 30.0 * nrisk

    return np.stack([f1, f2, f3], axis=1).mean(axis=0)


def evaluate_robust(x_unit: np.ndarray, engine: EQCAMScenarioEngine, run_seed: int,
                     n_batches: int = N_SCENARIO_BATCHES,
                     batch_size: int = SCENARIO_BATCH_SIZE,
                     beta: float = ROBUST_BETA,
                     batch_arrays_list=None) -> np.ndarray:
    """
    Robust min-max style evaluation of a single unit-cube candidate under
    compound uncertainty. Returns 3-vector of objectives to MINIMIZE.
    """
    x_phys = decode(x_unit)
    if batch_arrays_list is None:
        idx_batches = engine.draw_batches(n_batches, batch_size, run_seed)
        batch_arrays_list = [engine.batch_arrays(idx) for idx in idx_batches]
    per_batch = np.array([_simulate_batch(x_phys, b) for b in batch_arrays_list])  # (n_batches, 3)
    mean_f = per_batch.mean(axis=0)
    worst_f = per_batch.max(axis=0)
    return (1 - beta) * mean_f + beta * worst_f


def evaluate_population(X_unit: np.ndarray, engine: EQCAMScenarioEngine, run_seed: int,
                         n_batches: int = N_SCENARIO_BATCHES,
                         batch_size: int = SCENARIO_BATCH_SIZE) -> np.ndarray:
    """Evaluation over a population (n, DIM) -> (n, N_OBJ).
    Draws the compound-uncertainty scenario batches ONCE (cached, shared across the whole
    population) so every candidate in this call is judged on the same scenarios -- fair
    selection pressure -- and evaluation is fast (arrays sliced once, not per-candidate)."""
    idx_batches = engine.draw_batches(n_batches, batch_size, run_seed)
    batch_arrays_list = [engine.batch_arrays(idx) for idx in idx_batches]
    return np.array([
        evaluate_robust(x, engine, run_seed, n_batches, batch_size,
                         batch_arrays_list=batch_arrays_list)
        for x in X_unit
    ])
