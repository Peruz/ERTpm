import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from scipy.signal import detrend


def sinusoids_fft(sinusoids, injfreq, duration, sampling, hanning=True):
    """
    Find the twice the amplitude (i.e., max - min) of the measured voltage signals using the fft.

    injfreq: frequency of the injected current and thus of the voltage signal to analyze
    duration: duration of the signal in seconds
    sampling: sampling frequency

    Number of cycles (based on injfreq and duration) is too small (1 or 2),
    the hanning option should be set to false to avoid interference.

    The signal is real and so we can use the fftr.
    The sinusoids are detrended before the fft.
    Because of the Electra injection frequencies and design, it is ok to assume an even number of samples (sampleCount).
    With the fftr of a signal whose length is an even number:
    1. the frequency vector goes from 0 to the Nyquist frequency (fs / 2);
    2. the length of the frequency vector is the equal to the sampleCount divided (integer division, //) by 2 + 1.
    """
    sampleCount = int(round(duration * sampling))
    if sampleCount != sinusoids.shape[1]:
        raise ValueError('sampleCount and the length of the sinusoids have to be equal')
    if sampleCount % 2 != 0:
        raise ValueError('sampleCount has to be an even number')
    cycleCount = injfreq * duration
    if cycleCount % 2 != 0:
        raise ValueError('cycleCount is expected to be an even number')
    cycleCount = int(round(cycleCount))
    sampleCount_perCycle = int(round(sampleCount / injfreq))
    sinusoids = detrend(sinusoids, axis=1)
    if hanning:
        hann_window = np.hanning(sampleCount)
        sinusoids = sinusoids * hann_window
    ffta = np.fft.rfft(sinusoids, axis=1)
    ffta_magnitude = np.abs(ffta)
    ffta_magnitude /= sampleCount
    fftf_len = sampleCount // 2 + 1
    fftf = np.linspace(0, sampling / 2, fftf_len, dtype=int)
    injfreqIndex = np.where(fftf == cycleCount)[0][0]
    amp = ffta_magnitude[:, injfreqIndex]
    if hanning:
        amp *= 8  # because we need to recover the amp that we lost by hannning the signal
    else:
        amp *= 4  # because we only take the positive side and we want twice the amp (i.e., max - min)
    return(amp)


def sinfun_misfit(x, sinusoids, t, injfreq, hann_window):
    (amp, phase, mean) = x
    s = amp * np.sin((2 * np.pi * injfreq * t) + phase) + mean
    misfit = sinusoids - s
    misfit *= hann_window
    return(misfit)


def reject_outliers(sinusoids, m=1.5):
    d = np.abs(sinusoids - np.median(sinusoids))
    mdev = np.median(d)
    s = d / mdev if mdev else np.zero(len(d))
    return sinusoids[s < m]


def sinusoids_fit(sinusoids, injfreq, duration, sampling):
    """
    Find the twice the amplitude of the measured voltage signal (i.e, max - min)
    fitting it with a sinusoid of the same frequency.
    Use the hann window to adjust the weight of the initial and final points that may be noisy.
    Detrend the sinusoids before fitting the data.
    """
    t = np.linspace(0, duration, sampling * duration)
    amps = []
    hann_window = np.hanning(len(t))
    sinusoids = detrend(sinusoids, axis=1)
    for r in sinusoids:
        guess_mean = np.mean(r)
        mean_range = 1 + abs(guess_mean) / 10
        guess_amp = np.ptp(reject_outliers(r)) / 2
        amp_range = abs(guess_amp) / 10
        guess_phase = 0
        x0 = [guess_amp, guess_phase, guess_mean]
        bounds = (
            [guess_amp - amp_range, - 1, guess_mean - mean_range],
            [guess_amp + amp_range, + 1, guess_mean + mean_range],
        )
        sol = least_squares(
            sinfun_misfit,
            x0=x0,
            args=(r, t, injfreq, hann_window),
            jac='3-point',
            ftol=2.5e-15,
            xtol=2.5e-15,
            gtol=2.5e-15,
            loss='soft_l1',
            bounds=bounds,
            verbose=0,
        )
        amps.append(sol.x[0] * 2)
        (amp, phase, mean) = sol.x
        s = amp * np.sin((2 * np.pi * injfreq * t) + phase) + mean
    return(amps)
