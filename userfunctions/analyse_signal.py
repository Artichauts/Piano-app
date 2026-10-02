import numpy as np
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve



import os
import re
import numpy as np


def load_database(folder="Old_banque_donnees", pad_value=0.0, verbose=True):
    """
    Load every .npy file in `folder` and stack them into a 2D matrix
    of shape (n_signals, signal_length).

    - Files are sorted numerically by name (1.npy, 2.npy, ..., 10.npy, ...).
    - If signals have different lengths, shorter ones are padded
      with `pad_value` up to the longest one.

    Returns
    -------
    X : np.ndarray, shape (n_signals, max_length)
    names : list[str], file names in the same row order as X
    """
    files = [f for f in os.listdir(folder) if f.endswith(".npy")]
    files.sort(key=lambda f: int(re.findall(r"\d+", f)[0]))

    signals = [np.load(os.path.join(folder, f)).squeeze() for f in files]

    lengths = [len(s) for s in signals]
    max_len = max(lengths)

    if len(set(lengths)) > 1 and verbose:
        print(f"Warning: signals have different lengths "
              f"(min={min(lengths)}, max={max_len}), padding to {max_len}.")

    X = np.full((len(signals), max_len), pad_value, dtype=np.float32)
    for i, s in enumerate(signals):
        X[i, :len(s)] = s

    if verbose:
        print(f"Loaded {X.shape[0]} signals, each of length {X.shape[1]}")

    return X, files


def crop_after_max(X, N, include_peak=False, use_abs=False, pad_value=0.0):
    """
    For each signal, keep the N points that follow its maximum.

    Parameters
    ----------
    X : np.ndarray, shape (n_signals, signal_length), e.g. (297, 24000)
    N : int, number of points to keep after the max
    include_peak : bool
        If True, the window starts at the max itself (the peak is the first point).
        If False, it starts at the point right after the max.
    use_abs : bool
        If True, find the peak on |signal| (useful if the largest excursion is negative).
    pad_value : float
        Fill value if fewer than N points remain after the max.

    Returns
    -------
    out : np.ndarray, shape (n_signals, N)
    peaks : np.ndarray, shape (n_signals,), index of the max in each signal
    """
    n_signals, length = X.shape
    if N <= 0:
        raise ValueError("N must be positive")

    peaks = np.argmax(np.abs(X) if use_abs else X, axis=1)
    starts = peaks if include_peak else peaks + 1

    out = np.full((n_signals, N), pad_value, dtype=X.dtype)
    n_short = 0
    for i in range(n_signals):
        seg = X[i, starts[i]:starts[i] + N]
        out[i, :len(seg)] = seg
        n_short += len(seg) < N

    if n_short:
        print(f"Warning: {n_short} signal(s) had fewer than {N} points after the max "
              f"and were padded with {pad_value}.")

    return out, peaks






def corr_function(signal, matrice_centree, shape=(11, 27)):
    s = np.asarray(signal, dtype=float).ravel()                        # (T,)
    M = matrice_centree.astype(float)                                           # (N, T)
    # center both (a no-op for M if it is already centered)
    s = s - s.mean()
    M = M - M.mean(axis=1, keepdims=True)

    # Pearson r for every row against the signal -> (297,)
    num = M @ s
    den = np.linalg.norm(M, axis=1) * np.linalg.norm(s) + 1e-12
    correlations = num / den

    i_replique = int(np.argmax(correlations))
    corr = correlations[i_replique]                                    # best r (scalar)

    carte_corr = correlations.reshape(shape, order='F')                # (11, 27)

    return carte_corr, corr






if __name__ == "__main__":

    # Usage
    matrice, names = load_database("userfunctions/Old_banque_donnees")
    print(matrice.shape)
    
    # Usage
    matrice_after, peaks = crop_after_max(matrice, 2000, include_peak=True, use_abs=False)
    print(matrice_after.shape)  # (297, 5000)
    #print(tenseur_de_mesure.shape)
    #tenseur_crop =  extraire_apres_pic_max(tenseur_de_mesure, n_points=9000)
    #print("Jv pret à taper")
    #time.sleep(1)
    #print("Jv tape")

        
    #point_1 = record_audio(2, fs=48000)    
    #point_2 = record_audio(2, fs=48000)
    #point_3 = record_audio(2, fs=48000)

    #point_1_alligned = aligner_signal_sur_pic(point_1, indice_pic_cible)
    #point_2_alligned = aligner_signal_sur_pic(point_2, indice_pic_cible)
    #point_3_alligned = aligner_signal_sur_pic(point_3, indice_pic_cible)

    #print("Check")
    #plt.plot(tenseur_de_mesure[0, 224])
    #reference = tenseur_de_mesure[1, 224]
    #n_plot = min(len(reference), len(point_1_alligned))
    #plt.plot(reference[:n_plot], label="Référence")
    
    #plt.plot(point_1_alligned[:n_plot], label="Nouvelle mesure alignée")
    #plt.plot(point_2_alligned[:n_plot], label="Nouvelle mesure alignée")
    #plt.plot(point_3_alligned[:n_plot], label="Nouvelle mesure alignée")
    plt.plot(matrice_after[100])
    plt.plot(matrice_after[101])
    #plt.plot(matrice_after[32])
    plt.show()

    corr1_map, corr = corr_function(matrice_after[100], matrice_after)


    plt.imshow(corr1_map, cmap='coolwarm', vmin=-1, vmax=1)
    plt.show()


















def analyse_signal(signal, matrice, grid_shape=None):
    """
    signal : (n_samples,) or (n_samples, channels)
    matrice: (n_positions, n_samples_ref), one reference signal per row
    returns: corr_map (2D), note (str)
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim > 1:
        signal = signal[:, 0]                       # first channel only

    # Match the length of the reference signals (crop or zero-pad)
    n = matrice.shape[1]
    if len(signal) >= n:
        signal = signal[:n]
    else:
        signal = np.pad(signal, (0, n - len(signal)))

    # Pearson correlation of the signal with each row
    m = matrice - matrice.mean(axis=1, keepdims=True)
    s = signal - signal.mean()
    corr = (m @ s) / (np.linalg.norm(m, axis=1) * np.linalg.norm(s) + 1e-12)

    # Reshape the vector of positions into a 2D map for display
    if grid_shape is None:
        side = int(round(np.sqrt(len(corr))))
        grid_shape = (side, side)
    corr_map = corr.reshape(grid_shape)

    best = int(np.argmax(corr))
    note = f"best position #{best} (r = {corr[best]:.2f})"
    return corr_map, note