import numpy as np
from matplotlib import pyplot as plt

PULSES = 500_000
INTERVAL_NS = 50


def calc_detection_times(efficiency: float, lifetime_ns: int) -> np.ndarray:
    emission_duration = np.log(np.random.rand(PULSES)) * -lifetime_ns
    if efficiency == 1.0:
        return (np.arange(0, PULSES) * INTERVAL_NS) + emission_duration
    else:
        return (
            np.cumsum(
                np.floor(np.log(np.random.rand(PULSES)) / np.log(1 - efficiency)) + 1
            )
            * INTERVAL_NS
        ) + emission_duration


def split_detections(detection_times: np.ndarray) -> list[np.ndarray]:
    split_mask = np.random.rand(len(detection_times)) <= 0.5
    return [detection_times[split_mask], detection_times[~split_mask]]


def calc_taus(detectors: list[np.ndarray], half_window_ns: int) -> np.ndarray:
    starts = np.searchsorted(detectors[1], detectors[0] - half_window_ns, side="left")
    ends = np.searchsorted(detectors[1], detectors[0] + half_window_ns, side="right")
    ranges = ends - starts

    taus = np.empty(ranges.sum())
    taus_curr = 0
    for i, t_1 in enumerate(detectors[0]):
        taus[taus_curr : taus_curr + ranges[i]] = (
            t_1 - detectors[1][starts[i] : ends[i]]
        )
        taus_curr += ranges[i]

    return taus


def plot_coincidence_events(taus: np.ndarray, bins: int) -> None:
    plt.hist(taus, bins)
    plt.title("tau - coincidence events")
    plt.xlabel("tau")
    plt.ylabel("coincidence events")
    plt.show()


if __name__ == "__main__":
    first_emitter_times = calc_detection_times(0.9, 3)
    second_emitter_times = calc_detection_times(0.9, 3)
    detection_times = np.sort(np.concat([first_emitter_times, second_emitter_times]))
    detectors = split_detections(detection_times)
    taus = calc_taus(detectors, 1000)

    plot_coincidence_events(taus, 10000)
