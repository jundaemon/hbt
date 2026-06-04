import math
import sys
import time

import numpy as np
import polars as pl
from matplotlib import pyplot as plt


def t_calc(
    rng: np.random.Generator, exp_N: int, T_ns: float, eff: float, lifetime_ns: float
) -> np.ndarray:
    dur = np.log(rng.random(exp_N)) * -lifetime_ns
    if eff == 1.0:
        return (np.arange(0, exp_N) * T_ns) + dur
    else:
        return (
            np.cumsum(np.floor(np.log(rng.random(exp_N)) / np.log(1 - eff)) + 1) * T_ns
        ) + dur


def t_split(rng: np.random.Generator, set_t: np.ndarray) -> list[np.ndarray]:
    mask = rng.random(len(set_t)) <= 0.5
    return [set_t[mask], set_t[~mask]]


def tau_calc(sets_t: list[np.ndarray], half_window_ns: float) -> np.ndarray:
    starts = np.searchsorted(sets_t[1], sets_t[0] - half_window_ns, side="left")
    ranges = (
        np.searchsorted(sets_t[1], sets_t[0] + half_window_ns, side="right") - starts
    )

    return (
        np.repeat(sets_t[0], ranges)
        - sets_t[1][
            (
                np.repeat(starts, ranges)
                + np.arange(0, ranges.sum())
                - np.repeat(np.cumsum(ranges) - ranges, ranges)
            )
        ]
    )


def g2_zero_calc(taus: np.ndarray, T_ns: float, bins: int) -> np.float64:
    hist, edges = np.histogram(taus, bins)
    bins_per_pulse = math.floor(T_ns / (edges[1] - edges[0]))

    side_crests_i = (
        np.where(
            (hist[1:-1] > hist[2:])
            & (hist[1:-1] > hist[:-2])
            & (hist[1:-1] > hist.max() * 0.9)
        )[0]
        + 1
    )
    side_crests_i = side_crests_i[
        np.insert(np.diff(side_crests_i) > math.floor(bins_per_pulse * 0.9), 0, True)
    ]

    mask = np.full(len(side_crests_i), True)
    mask[[0, -1]] = False
    side_crests_i = side_crests_i[mask]

    areas = np.empty(len(side_crests_i))
    for i, side_trough_i in enumerate(side_crests_i - bins_per_pulse // 2):
        areas[i] = hist[side_trough_i : side_trough_i + bins_per_pulse].sum()

    zero_trough_i = len(hist) // 2 - bins_per_pulse // 2
    return hist[zero_trough_i : zero_trough_i + bins_per_pulse].sum() / areas.mean()


def plot_coincidence_events(taus: np.ndarray, bins: int) -> None:
    plt.hist(taus, bins)
    plt.title("tau - coincidence events")
    plt.xlabel("tau")
    plt.ylabel("coincidence events")
    plt.show()


EXP_N = 500_000
T_NS = 50.0
LIFETIME_NS = 3.0
HALF_WINDOW_NS = 1_000.0
BINS = 10_000
SEED = 10
RNG = np.random.default_rng(SEED)


def run_generation() -> None:
    effs = np.linspace(0.1, 1.0, 10, True)
    data_points = len(effs) ** 2
    results = {
        "Seed used": np.full(data_points, SEED, np.int64),
        "Lifetime (ns)": np.full(data_points, LIFETIME_NS, np.float64),
        "Time between pulses (ns)": np.full(data_points, T_NS, np.float64),
        "Efficiency of medium 1": np.empty(data_points, np.float64),
        "Efficiency of medium 2": np.empty(data_points, np.float64),
        "Detections at detector 1": np.empty(data_points, np.int64),
        "Detections at detector 2": np.empty(data_points, np.int64),
        "g2": np.empty(data_points, np.float64),
    }

    idx = 0
    total_time_s = 0
    print("Started calculations\n")

    for eff_1 in effs:
        for eff_2 in effs:
            start = time.perf_counter()

            set_t_1 = t_calc(RNG, EXP_N, T_NS, eff_1, LIFETIME_NS)
            set_t_2 = t_calc(RNG, EXP_N, T_NS, eff_2, LIFETIME_NS)

            set_t = np.sort(np.concat([set_t_1, set_t_2]))
            sets_t = t_split(RNG, set_t)

            taus = tau_calc(sets_t, HALF_WINDOW_NS)
            g2_zero = g2_zero_calc(taus, T_NS, BINS)

            end = time.perf_counter()
            print(f"Finished a calculation, time taken: {(end - start):.2f}s")
            total_time_s += end - start

            results["g2"][idx] = g2_zero
            results["Efficiency of medium 1"][idx] = eff_1
            results["Efficiency of medium 2"][idx] = eff_2
            results["Detections at detector 1"][idx] = len(sets_t[0])
            results["Detections at detector 2"][idx] = len(sets_t[1])

            idx += 1

    print(f"\nFinished {data_points} calculations")
    print(f"Total time taken: {total_time_s:.2f}s")

    pl.DataFrame(results).write_excel("results.xlsx", autofit=True)
    print("\nExcel sheet generated")


if __name__ == "__main__":
    try:
        run_generation()
    except Exception as e:
        print(e)
        sys.exit(1)
