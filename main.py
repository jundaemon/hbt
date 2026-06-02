import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks


def t_calc(
    rng: np.random.Generator, exp_N: int, T_ns: int, eff: float, lifetime_ns: int
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


def tau_calc(sets_t: list[np.ndarray], half_window_ns: int) -> np.ndarray:
    starts = np.searchsorted(sets_t[1], sets_t[0] - half_window_ns, side="left")
    ranges = (
        np.searchsorted(sets_t[1], sets_t[0] + half_window_ns, side="right") - starts
    )

    return (
        np.repeat(sets_t[0], ranges)
        - sets_t[1][
            (
                np.repeat(starts, ranges)
                + np.arange(ranges.sum())
                - np.repeat(np.cumsum(ranges) - ranges, ranges)
            )
        ]
    )


def g2_zero_calc(taus: np.ndarray, T_ns: int, bins: int) -> np.float64:
    hist, edges = np.histogram(taus, bins)
    bins_per_pulse = int(T_ns / (edges[1] - edges[0]))

    side_crests_i, _ = find_peaks(
        hist,
        distance=int(bins_per_pulse * 0.9),
        prominence=hist.max() * 0.8,
    )
    mask = np.ones(len(side_crests_i), bool)
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


if __name__ == "__main__":
    exp_N = 500_000
    T_ns = 50
    eff = 1
    lifetime_ns = 3
    half_window_ns = 1000
    bins = 10000

    rng = np.random.default_rng(10)

    start = time.perf_counter()

    set_t_1 = t_calc(rng, exp_N, T_ns, eff, lifetime_ns)
    set_t_2 = t_calc(rng, exp_N, T_ns, eff, lifetime_ns)

    set_t = np.sort(np.concat([set_t_1, set_t_2]))
    sets_t = t_split(rng, set_t)

    taus = tau_calc(sets_t, half_window_ns)
    g2_zero = g2_zero_calc(taus, T_ns, bins)

    end = time.perf_counter()

    print(f"Calculated g2: {g2_zero}")
    print(f"Time taken: {end - start}")
