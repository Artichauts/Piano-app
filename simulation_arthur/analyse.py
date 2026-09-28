import numpy as np
import matplotlib.pyplot as plt

def cnr_2d(map2d):
    """
    Calcule le CNR (Contrast-to-Noise Ratio) d'une carte de corrélation 2D.
    Signal = région >= half_max (le lobe principal)
    Bruit  = reste de la carte (arrière-plan)
    CNR = (mean_signal - mean_background) / std_background
    """
    max_val = np.max(map2d)
    half_max = max_val / 2.0

    signal_mask = map2d >= half_max
    background_mask = ~signal_mask

    if np.sum(background_mask) == 0 or np.sum(signal_mask) == 0:
        return 0.0

    mean_signal = np.mean(map2d[signal_mask])
    mean_background = np.mean(map2d[background_mask])
    std_background = np.std(map2d[background_mask])

    if std_background == 0:
        return 0.0

    cnr = (mean_signal - mean_background) / std_background
    return cnr


def cnr_map_2d(corr_tensor, N, M):
    """
    Calcule le CNR pour chaque point (i,j) et retourne une map (N x M).
    """
    cnr_map = np.zeros((N, M))

    for i in range(N):
        for j in range(M):
            map2d = corr_tensor[i, j, :, :]
            cnr_map[i, j] = cnr_2d(map2d)

    return cnr_map



def fwhm_2d_area(map2d):
    """
    Calcule un FWHM 2D basé sur l'aire de la région >= half_max,
    converti en diamètre équivalent (aire d'un disque) en cm.
    """
    max_val = np.max(map2d)
    half_max = max_val / 2.0
    
    above = map2d >= half_max
    area = np.sum(above)  # nombre de pixels au-dessus de half_max
    
    if area == 0:
        return 0.0
    
    # Diamètre équivalent d'un disque de même aire : A = pi*(d/2)^2
    diameter = 2 * np.sqrt(area / np.pi) * 0.5
    return diameter


def fwhm_map_2d(corr_tensor, N, M):
    """
    Calcule le FWHM 2D pour chaque point (i,j) et retourne une map (N x M).
    """
    fwhm_map = np.zeros((N, M))
    
    for i in range(N):
        for j in range(M):
            map2d = corr_tensor[i, j, :, :]
            fwhm_map[i, j] = fwhm_2d_area(map2d)
    
    return fwhm_map







# Traces: shape (N, M, Nt, n_sensors) = (82, 15, 2000, 7), float32
sim = np.load(".\\simulations\\sim_shapeB_steel.npy", mmap_mode="r")   # lazy, doesn't load into RAM
print(sim.shape, sim.dtype)

# Metadata
meta = np.load(".\\simulations\\sim_shapeB_steel_meta.npz")
print(meta.keys)
#src_ix, src_iy = meta["src_ix"], meta["src_iy"]
#sensor_pts     = meta["sensor_pts"]
N, M = int(meta["N"]), int(meta["M"])
dt, dx    = float(meta["dt"]), float(meta["dx"]), #int(meta["Nt"])
c_plate        = float(meta["c_plate"])
plate_mask     = meta["plate_mask"]

N_sensor = 7
Nt = 9824
print(dt)



corr_tensor = np.zeros((N_sensor, N, M, N, M))

mean_fwhm = np.zeros((N_sensor))
mean_fwhm_region = np.zeros((N_sensor))

std_fwhm = np.zeros((N_sensor))
std_fwhm_region = np.zeros((N_sensor))


mean_cnr = np.zeros((N_sensor))
mean_cnr_region = np.zeros((N_sensor))

std_cnr = np.zeros((N_sensor))
std_cnr_region = np.zeros((N_sensor))



mask = np.zeros((N, M), dtype=bool)
mask[40:104, 2:32] = True 

# MASK DE ROI
r0, r1 = 40, 104
c0, c1 = 0, 30
Nc, Mc = r1 - r0, c1 - c0

sim = sim.reshape(N_sensor, Nt, N, M, order='F')   
sim_c = sim[:, :, r0:r1, c0:c1]

corr_tensor = np.zeros((N_sensor, Nc, Mc, Nc, Mc))

for i in range(N_sensor):
    X = sim_c[i].reshape(Nt, Nc * Mc)
    corr_matrix = np.nan_to_num(np.corrcoef(X.T), nan=0.0).reshape(Nc, Mc, Nc, Mc)
    corr_tensor[i] = corr_matrix

    fwhm = fwhm_map_2d(corr_matrix, Nc, Mc) # Résolution
    cnr = cnr_map_2d(corr_matrix, Nc, Mc)   # Contraste

    mean_fwhm[i] = np.mean(fwhm)
    std_fwhm[i] = np.std(fwhm)

    mean_cnr[i] = np.mean(cnr)
    std_cnr[i] = np.std(cnr)

print(f"resolution de ({np.mean(mean_fwhm)} pm {np.linalg.norm(std_fwhm)/np.sqrt(7)}) cm sur l'ensemble de la plate")
print(f"Contraste de ({np.mean(mean_cnr)} pm {np.linalg.norm(std_cnr)/np.sqrt(7)}) cm sur l'ensemble de la plate")

print(mean_fwhm)


i0, j0 = 20, 20
matrix = corr_tensor[5]
correlation = matrix[i0, j0, :, :]  # forme (N, M)
fwhm = fwhm_map_2d(matrix, Nc, Mc)
cnr = cnr_map_2d(matrix, Nc, Mc)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 5), sharey=True)

# --- Corrélation ---
im1 = ax1.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title(f'Corrélation pour le point (i={i0}, j={j0})')
cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('Corrélation')

# --- Résolution (FWHM) ---
im2 = ax2.imshow(fwhm, cmap='Spectral')
ax2.set_xlabel("x")
ax2.set_title('Résolution')
cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('FWHM')

# --- Contraste (CNR) ---
im3 = ax3.imshow(cnr, cmap='Spectral')
ax3.set_xlabel("x")
ax3.set_title('Contraste')
cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
cbar3.set_label('CNR')

plt.tight_layout()
plt.show()