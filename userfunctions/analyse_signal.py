import numpy as np

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