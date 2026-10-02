import numpy as np
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt

from RecordMicro import record_audio

def extraire_apres_pic_max(tenseur, n_points=9000):
    """Garde n_points à partir du pic absolu de chaque signal d'un tenseur 3D."""
    tenseur = np.asarray(tenseur)
    if tenseur.ndim != 3:
        raise ValueError("Le tenseur doit avoir la forme (répliques, points, échantillons).")
    if n_points <= 0:
        raise ValueError("n_points doit être supérieur à zéro.")
    if tenseur.shape[2] == 0:
        raise ValueError("Les signaux ne peuvent pas être vides.")

    indices_pics = np.argmax(tenseur, axis=2)
    extraits = np.zeros((*tenseur.shape[:2], n_points), dtype=tenseur.dtype)
    for i_replique, i_point in np.ndindex(tenseur.shape[:2]):
        debut = indices_pics[i_replique, i_point]
        fin = min(debut + n_points, tenseur.shape[2])
        extrait = tenseur[i_replique, i_point, debut:fin]
        extraits[i_replique, i_point, :len(extrait)] = extrait

    return extraits


def premier_pic(signal, seuil_relatif=0.06):
    amplitude = np.abs(signal - signal[0])
    seuil = amplitude.max() * seuil_relatif
    pics = np.flatnonzero(
        (amplitude[1:-1] >= amplitude[:-2])
        & (amplitude[1:-1] > amplitude[2:])
        & (amplitude[1:-1] >= seuil)
    ) + 1
    return int(pics[0]) if pics.size else int(np.argmax(amplitude))




def aligner_signal_sur_pic(signal, indice_cible):
    signal = np.asarray(signal, dtype=float).reshape(-1)
    signal = signal - signal[0]
    decalage = indice_cible - premier_pic(signal)
    if decalage >= 0:
        return np.pad(signal, (decalage, 0))[:len(signal)]
    return np.pad(signal[-decalage:], (0, -decalage))


def aligner_matrice_sur_premier_pic(matrice):
    indices_pics = np.array([
        [premier_pic(signal) for signal in replique]
        for replique in matrice
    ])
    indice_cible = int(indices_pics.max())
    matrice_alignee = np.empty_like(matrice)
    for i_replique in range(matrice.shape[0]):
        for i_point in range(matrice.shape[1]):
            signal = matrice[i_replique, i_point] - matrice[i_replique, i_point, 0]
            decalage = indice_cible - indices_pics[i_replique, i_point]
            matrice_alignee[i_replique, i_point] = np.pad(
                signal, (decalage, 0)
            )[:matrice.shape[2]]
    return matrice_alignee, indice_cible



def charger_banque(dossier, N_LIGNES = 11, N_COLONNES = 27):
    """Charge les points triés; les répliques complètes sont moyennées."""
    groupes = {}
    for fichier in Path(dossier).glob("*.npy"):
        morceaux = fichier.stem.split(".")
        if len(morceaux) == 1 and morceaux[0].isdigit():
            numero, replique = int(morceaux[0]), 0
        elif (len(morceaux) == 2 and all(part.isdigit() for part in morceaux)
              and 1 <= int(morceaux[1]) <= 3):
            numero, replique = int(morceaux[0]), int(morceaux[1])
        else:
            continue
        if 1 <= numero <= N_LIGNES * N_COLONNES:
            groupes.setdefault(numero, {})[replique] = fichier

    numeros = []
    signaux = []
    for numero, fichiers in sorted(groupes.items()):
        if all(replique in fichiers for replique in (1, 2, 3)):
            repliques = [np.load(fichiers[replique]) for replique in (1, 2, 3)]
        elif 0 in fichiers:
            signal = np.load(fichiers[0])
            repliques = [signal, signal, signal]
        else:
            continue
        numeros.append(numero)
        signaux.append(repliques)

    if not signaux:
        raise ValueError("Aucun point complet trouvé dans la banque de données.")

    n = min(len(signal) for repliques in signaux for signal in repliques)
    matrice = np.stack([
        [repliques[i][:n] for repliques in signaux]
        for i in range(3)
    ])
    print(matrice.shape)
    matrice, indice_pic_cible = aligner_matrice_sur_premier_pic(matrice)
    matrice_centree = matrice - matrice.mean(axis=2, keepdims=True)


    #positions = np.array([case_vers_position(k) for k in numeros])
    return matrice_centree, indice_pic_cible


def corr_function(signal, matrice_centree, debut=0, fin=None):
        signal = np.asarray(signal, dtype=float)
        if signal.ndim > 1:
            signal = signal[:, 0]

        matrice_centree = np.asarray(matrice_centree)
        if matrice_centree.ndim == 2:
            matrice_centree = matrice_centree[np.newaxis, :, :]

        if fin is None:
            fin = matrice_centree.shape[2]
        signal = signal[debut:fin]
        matrice_centree = matrice_centree[:, :, debut:fin]

        n = matrice_centree.shape[2]
        if len(signal) != n:
            signal = signal[:n] if len(signal) > n else np.pad(signal, (0, n - len(signal)))

        s = signal - signal.mean()
        norm = np.linalg.norm(matrice_centree, axis=2)
        correlations = (matrice_centree @ s) / (norm * np.linalg.norm(s) + 1e-12)
        i_replique = np.unravel_index(np.argmax(correlations), correlations.shape)[0]
        corr = correlations[i_replique]

        print(corr.shape)
        carte_corr = corr.reshape(11, 27, order='F')
        #carte_corr = np.full((N_LIGNES, N_COLONNES), np.nan)
    
        return carte_corr, corr






if __name__ == "__main__":
    DOSSIER = Path("Piano\\Piano-app\\userfunctions\\banque_donnees")
    tenseur_de_mesure, indice_pic_cible = charger_banque(DOSSIER)
    print(tenseur_de_mesure.shape)
    #tenseur_crop =  extraire_apres_pic_max(tenseur_de_mesure, n_points=9000)
    print("Jv pret à taper")
    time.sleep(1)
    print("Jv tape")

        
    point_1 = record_audio(2, fs=48000)
    point_2 = record_audio(2, fs=48000)
    point_3 = record_audio(2, fs=48000)

    point_1_alligned = aligner_signal_sur_pic(point_1, indice_pic_cible)
    point_2_alligned = aligner_signal_sur_pic(point_2, indice_pic_cible)
    point_3_alligned = aligner_signal_sur_pic(point_3, indice_pic_cible)

    print("Check")
    plt.plot(tenseur_de_mesure[0, 224])
    reference = tenseur_de_mesure[1, 224]
    n_plot = min(len(reference), len(point_1_alligned))
    plt.plot(reference[:n_plot], label="Référence")
    
    plt.plot(point_1_alligned[:n_plot], label="Nouvelle mesure alignée")
    plt.plot(point_2_alligned[:n_plot], label="Nouvelle mesure alignée")
    plt.plot(point_3_alligned[:n_plot], label="Nouvelle mesure alignée")

    plt.legend()
    plt.show()

    corr1_map, corr = corr_function(point_1_alligned, tenseur_de_mesure[2,:])
    corr2_map, corr = corr_function(point_1_alligned, tenseur_de_mesure[1,:])
    corr3_map, corr = corr_function(point_1_alligned, tenseur_de_mesure[1,:])

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