## Introduction
This is a program that runs a Monte Carlo simulation of the [Hanbury Brown & Twiss experiment](https://en.wikipedia.org/wiki/Hanbury_Brown_and_Twiss_effect). It also doubles as a synthetic data generator for the second order correlation function which will be used in my final year project.

## Prerequisites
- Have a [Python](https://www.python.org/downloads/) interpreter installed, code works on version 3.14.
- Have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed.

## Usage
Clone the repository
```
git clone https://github.com/jundaemon/hbt.git
```

Create a virtual environment
```
uv venv
```

Then install dependencies (no need to activate the virtual environment, uv does that for you for any uv command)
```
uv sync --frozen
```

Run the program and you should get an excel sheet with generated values
```
uv run main.py
```

## Core calculations walkthrough
```python
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
```

This function calculates a series of timestamps when photons are detected at a detector, assuming emission to detection is instant. An emitter will emit `exp_N` (expected N) photons and the time at which they are detected is affected by `T_ns` (period of pulses), `eff` (efficiency of the medium ,$[0, 1)$) and `lifetime_ns` (lifetime of electrons).

Photons are not immediately emitted when the medium is pulsed with a laser. Instead, the time taken for photon emission is modelled by exponential distribution, $P(t) = e^{-\frac{t}{\tau}}$, where $\tau$ is the lifetime of an electron. Solving for $t$, $t = -\tau \cdot \ln(U)$, where $U$ is arbitrary in the range $[0, 1)$. The first line of this function thus calculates the time taken for photon emission after a laser pulse for all photons.

Photons are not necassarily guaranteed to be emitted after a laser pulse, this is instead determined by the efficiency of the medium. If the efficiency of the medium is 1 in the range $(0, 1]$, then photons are guaranteed to be emitted. We thus add the emission duration to the period of the pulse multiplied with the index of the photon for all photons and return it. But if the efficiency of the medium is less than 1, then the emission of a photon is not guaranteed for each pulse. The naive way to handle this would be to loop until the set is full and for each iteration, compare a randomly generated number with the efficiency of the medium, if it is lower, add the timestamp to the set and continue. Such an approach would look like

```python
set_t = np.empty(exp_N)
i = 0
while i < exp_N:
    if rng.random() < eff:
        set_t[i] = i * T_ns
        i += 1
```

This is terribly slow, we are iterating more than $n$ times for an $n$ sized array, not to mention the generation of a random number for each iteration. To combat this, we use a geometric distribution to determine the number of pulses needed for a successful photon emission, based on the efficiency of the medium. The number of pulses needed for a successful photon emission is modelled by $P(k) = 1 - (1 - eff)^k$. Solving for $k$, $k = \frac{\ln(1 - U)}{\ln(1 - eff)}$, where $U$ is arbitrary in range $[0, 1)$. `np.cumsum(np.floor(np.log(rng.random(exp_N)) / np.log(1 - eff)) + 1)` thus calculates the indexes where a pulse leads to a successful photon emission. The result is then multiplied with the period of pulse, the duration of emission is then added at the end.
