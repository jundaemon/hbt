import time

import numpy as np
from matplotlib import pyplot as plt

exp_N = 500_000
T_ns = 50


def t_calc(eff: float, lifetime_ns: int) -> np.ndarray:
    dur = np.log(np.random.rand(exp_N)) * -lifetime_ns
    if eff == 1.0:
        return (np.arange(0, exp_N) * T_ns) + dur
    else:
        return (
            np.cumsum(np.floor(np.log(np.random.rand(exp_N)) / np.log(1 - eff)) + 1)
            * T_ns
        ) + dur


def t_split(set_t: np.ndarray) -> list[np.ndarray]:
    mask = np.random.rand(len(set_t)) <= 0.5
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


def plot_coincidence_events(taus: np.ndarray, bins: int) -> None:
    plt.hist(taus, bins)
    plt.title("tau - coincidence events")
    plt.xlabel("tau")
    plt.ylabel("coincidence events")
    plt.show()


if __name__ == "__main__":
    start = time.perf_counter()
    set_t_1 = t_calc(0.1, 3)
    set_t_2 = t_calc(0.1, 3)
    set_t = np.sort(np.concat([set_t_1, set_t_2]))
    sets_t = t_split(set_t)
    taus = tau_calc(sets_t, 1000)
    end = time.perf_counter()

    print(end - start)
