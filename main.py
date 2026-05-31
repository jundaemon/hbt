import numpy as np
from matplotlib import pyplot as plt

EXP_N = 500_000
T_NS = 50


def t_calc(eff: float, tau_ns: int) -> np.ndarray:
    dur = np.log(np.random.rand(EXP_N)) * -tau_ns
    if eff == 1.0:
        return (np.arange(0, EXP_N) * T_NS) + dur
    else:
        return (
            np.cumsum(np.floor(np.log(np.random.rand(EXP_N)) / np.log(1 - eff)) + 1)
            * T_NS
        ) + dur


def t_split(set_t: np.ndarray) -> list[np.ndarray]:
    mask = np.random.rand(len(set_t)) <= 0.5
    return [set_t[mask], set_t[~mask]]


def tau_calc(sets_t: list[np.ndarray], half_window_ns: int) -> np.ndarray:
    starts = np.searchsorted(sets_t[1], sets_t[0] - half_window_ns, side="left")
    ends = np.searchsorted(sets_t[1], sets_t[0] + half_window_ns, side="right")
    ranges = ends - starts

    taus = np.empty(ranges.sum())
    taus_i = 0
    for i, t_1 in enumerate(sets_t[0]):
        taus[taus_i : taus_i + ranges[i]] = t_1 - sets_t[1][starts[i] : ends[i]]
        taus_i += ranges[i]

    return taus


def plot_coincidence_events(taus: np.ndarray, bins: int) -> None:
    plt.hist(taus, bins)
    plt.title("tau - coincidence events")
    plt.xlabel("tau")
    plt.ylabel("coincidence events")
    plt.show()


if __name__ == "__main__":
    set_t_1 = t_calc(0.1, 3)
    set_t_2 = t_calc(0.1, 3)
    set_t = np.sort(np.concat([set_t_1, set_t_2]))
    sets_t = t_split(set_t)
    taus = tau_calc(sets_t, 1000)

    plot_coincidence_events(taus, 10000)
