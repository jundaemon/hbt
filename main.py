import numpy as np
from matplotlib import pyplot as plt


def calc_detection_times(
    pulses: int, efficiency: float, lifetime_ns: int, interval_ns: int
) -> np.ndarray:
    # on each pulse, the probability of emitting a photon is modelled by efficiency
    # this just returns the 0 indexed indices of pulses that led to emittance
    pulse_times = np.flatnonzero(np.random.rand(pulses) < efficiency) * interval_ns

    # the time at which photons are emitted is modelled by exponential decay
    emission_times = np.log(np.random.rand(len(pulse_times))) * -lifetime_ns
    return pulse_times + emission_times


def split_detections(detection_times: np.ndarray) -> list[np.ndarray]:
    split_mask = np.random.rand(len(detection_times)) <= 0.5
    return [detection_times[split_mask], detection_times[~split_mask]]


def calc_taus(detectors: list[np.ndarray], half_window_ns: int) -> np.ndarray:
    # preallocating a generous array so the underlying array doesn't need to constantly resize
    taus = np.zeros(len(detectors[0]) ** 2)
    curr_len = 0
    for t_a in detectors[0]:
        start = np.searchsorted(detectors[1], t_a - half_window_ns)
        end = np.searchsorted(detectors[1], t_a + half_window_ns)

        for t_b in detectors[1][start : end + 1]:
            taus[curr_len] = t_a - t_b
            curr_len += 1

    return taus[:curr_len]


def plot_coincidence_events(taus: np.ndarray, bins: int) -> None:
    plt.hist(taus, bins)
    plt.title("tau - coincidence events")
    plt.xlabel("tau")
    plt.ylabel("coincidence events")
    plt.show()


if __name__ == "__main__":
    """
    # single photon emission at 90% efficiency, 10ns lifetime and 100ns time between pulse
    detection_times = calc_detection_times(10000, 0.90, 10, 100)
    detectors = split_detections(detection_times)
    taus = calc_taus(detectors, 500)
    plot_coincidence_events(taus, 500)
    """

    # two single photon emitters at 90% efficiency, 10ns lifetime and 100ns time between pulse
    first_emitter_times = calc_detection_times(5000, 0.90, 10, 100)
    second_emitter_times = calc_detection_times(5000, 0.90, 10, 100)
    detection_times = np.sort(np.concat([first_emitter_times, second_emitter_times]))
    detectors = split_detections(detection_times)
    taus = calc_taus(detectors, 500)
    plot_coincidence_events(taus, 500)
