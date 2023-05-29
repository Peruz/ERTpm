import numpy as np
from ertsinusoids import fit_sin
# import matplotlib.pyplot as plt

amp = 1
injfreq = 8
phase = 0
mean = 0

duration = 1
sampling = 256
t = np.linspace(0, duration, duration * sampling)

s = amp * np.sin((2 * np.pi * injfreq * t) + phase) + mean
s = np.tile(s, (2, 1))

a = fit_sin(s, t, 8)

print(a)
