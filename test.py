import numpy as np, matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

fs = 48000
t = np.arange(0, 0.05, 1/fs)
x = np.sin(2*np.pi*500*t) + 0.5*np.sin(2*np.pi*3000*t)
sos = butter(4, 1000, btype="low", fs=fs, output="sos")
y = sosfiltfilt(sos, x)                                   # passe-bas à 1000 Hz

f = np.fft.rfftfreq(len(x), 1/fs)
fig, ax = plt.subplots(2, 2, figsize=(10, 6))
ax[0, 0].plot(t*1000, x);  ax[0, 0].set_title("Signal original")
ax[1, 0].plot(t*1000, y);  ax[1, 0].set_title("Après filtre passe-bas 1000 Hz")
ax[0, 1].plot(f, np.abs(np.fft.rfft(x))); ax[0, 1].set_title("FFT de l'original")
ax[1, 1].plot(f, np.abs(np.fft.rfft(y))); ax[1, 1].set_title("FFT du signal filtré")
for a in ax[:, 1]: a.set_xlim(0, 5000)
plt.show()