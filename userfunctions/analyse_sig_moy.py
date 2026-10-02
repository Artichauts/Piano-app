import numpy as np
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


N_LIGNES = 5
N_COLONNES = 14

plage = {
    "do": [(1, 3), (6, 8)],
    "do#": [(4, 5), (9, 10), (14, 15)],
    "ré": [(11, 13), (16, 18)],
    "ré#": [(19, 20), (24, 25), (29, 30)],
    "mi": [(21, 23), (26, 28)],
    "fa": [(31, 33), (36, 38)],
    "fa#": [(34, 35), (39, 40), (44, 45)],
    "sol": [(41, 43), (46, 48)],
    "sol#": [(49, 50), (54, 55)],
    "la": [(51, 53), (56, 58)],
    "la#": [(59, 60), (64, 65), (69, 70)],
    "si": [(61, 63), (66, 68)],
}

# plage = {
#     "do": [(1, 38)],
#     "do#": [(39, 44), (50, 55), (61, 66)],
#     "ré": [(45, 49), (56, 60), (67, 77), (78, 82)],
#     "ré#": [(83, 88), (94, 99)],
#     "mi": [(89, 93), (100, 132)],
#     "fa": [(133, 159)],
#     "fa#": [(160, 165), (171, 176), (182, 187)],
#     "sol": [(166, 170), (177, 181), (188, 203)],
#     "sol#": [(204, 209), (215, 220), (226, 231)],
#     "la": [(210, 214), (221, 225), (232, 247)],
#     "la#": [(248, 253), (259, 264), (270, 275)],
#     "si": [(254, 258), (265, 269), (276, 297)],
#     }

def case_vers_position(k):
    """Case 1-70 vers (ligne, colonne), avec la ligne 0 en haut."""
    return N_LIGNES - 1 - (k - 1) % N_LIGNES, (k - 1) // N_LIGNES


def construire_table_notes(plages):
    """
    plages : dict note -> liste d'intervalles (debut, fin) inclusifs de numéros de case, ex :
        {"do": [(1, 38)], "do#": [(39, 44), (50, 55), (61, 66)], ...}
    retourne : dict {numero_case: note}
    """
    table = {}
    for note, intervalles in plages.items():
        for debut, fin in intervalles:
            for k in range(debut, fin + 1):
                table[k] = note
    return table


def charger_banque(dossier):
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
    matrice, indice_pic_cible = aligner_matrice_sur_premier_pic(matrice)
    matrice_centree = matrice - matrice.mean(axis=2, keepdims=True)
    normes = np.linalg.norm(matrice_centree, axis=2)

    positions = np.array([case_vers_position(k) for k in numeros])
    return matrice_centree, normes, numeros, positions, indice_pic_cible

"""
def analyser_signal(signal, matrice_centree, normes, numeros, positions, table_notes,
                    indice_pic_cible=None):
    Retourne la note, la carte de corrélation 2D, et le vecteur brut de corrélations.
    signal = np.asarray(signal, dtype=float)
    if signal.ndim > 1:
        signal = signal[:, 0]

    n = matrice_centree.shape[2]
    if len(signal) != n:
        signal = signal[:n] if len(signal) > n else np.pad(signal, (0, n - len(signal)))
    if indice_pic_cible is not None:
        signal = aligner_signal_sur_pic(signal, indice_pic_cible)

    s = signal - signal.mean()
    corr = (matrice_centree @ s) / (normes * np.linalg.norm(s) + 1e-12)

    carte_corr = np.full((N_LIGNES, N_COLONNES), np.nan)
    carte_corr[positions[:, 0], positions[:, 1]] = corr

    i_best = int(np.argmax(corr))
    note = table_notes.get(numeros[i_best])

    return note, carte_corr, corr
"""

def trouver_declenchement(M, seuil):
    """Indice du premier dépassement du seuil pour chaque ligne de M (2D).
    Si une ligne ne dépasse jamais le seuil, on prend le max de |signal|."""
    masque = np.abs(M) > seuil
    i_trig = masque.argmax(axis=1)
    sans_depassement = ~masque.any(axis=1)
    i_trig[sans_depassement] = np.abs(M[sans_depassement]).argmax(axis=1)
    return i_trig


def recadrer_matrice(matrice, seuil, n_avant, n_apres):
    """Recadre chaque ligne autour de son point de déclenchement
    (n_avant avant, n_apres après, zéros si on dépasse les bords)."""
    M = np.asarray(matrice, dtype=float)
    i_trig = trouver_declenchement(M, seuil)

    indices = i_trig[:, None] + np.arange(-n_avant, n_apres)[None, :]
    valide = (indices >= 0) & (indices < M.shape[1])

    lignes = np.arange(M.shape[0])[:, None]
    sortie = M[lignes, np.clip(indices, 0, M.shape[1] - 1)]
    sortie[~valide] = 0.0
    return sortie


def preparer_matrice(matrice, seuil, n_avant, n_apres):
    """À faire UNE SEULE FOIS au chargement : recadre, centre, calcule les normes."""
    M = recadrer_matrice(matrice, seuil, n_avant, n_apres)
    M = M - M.mean(axis=1, keepdims=True)
    normes = np.linalg.norm(M, axis=1)
    return M, normes


def analyser_signal(signal, matrice_centree, normes, numeros, positions, table_notes,
                    indice_pic_cible=None):
    """Retourne la note au coefficient moyen maximal et la carte moyennée."""
    signal = np.asarray(signal, dtype=float)
    if signal.ndim > 1:
        signal = signal[:, 0]

    matrice_centree = np.asarray(matrice_centree)
    normes = np.asarray(normes)
    if matrice_centree.ndim == 2:
        matrice_centree = matrice_centree[np.newaxis, :, :]
        normes = normes[np.newaxis, :]

    n = matrice_centree.shape[2]
    if len(signal) != n:
        signal = signal[:n] if len(signal) > n else np.pad(signal, (0, n - len(signal)))
    if indice_pic_cible is not None:
        signal = aligner_signal_sur_pic(signal, indice_pic_cible)

    s = signal - signal.mean()
    corr = (np.abs(matrice_centree) @ np.abs(s)) / (normes * np.linalg.norm(s) + 1e-12)
    corr_cases = corr.mean(axis=0)

    indices_par_note = {}
    for i_case, numero in enumerate(numeros):
        note_case = table_notes.get(numero)
        if note_case is not None:
            indices_par_note.setdefault(note_case, []).append(i_case)
    moyennes_notes = {
        note: corr_cases[indices].mean()
        for note, indices in indices_par_note.items()
    }
    if not moyennes_notes:
        raise ValueError("Aucune case de la matrice n'est associée à une note.")
    note = max(moyennes_notes, key=moyennes_notes.get)

    carte_corr = np.full((N_LIGNES, N_COLONNES), np.nan)
    carte_corr[positions[:, 0], positions[:, 1]] = corr_cases

    return note, carte_corr, corr_cases

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
        carte_corr = corr.reshape(N_LIGNES, N_COLONNES, order='F')[::-1]
        #carte_corr = np.full((N_LIGNES, N_COLONNES), np.nan)
    
        return carte_corr, corr


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


if __name__ == "__main__":
    #load matrice banque de données
    DOSSIER = Path("Piano\\Piano-app\\userfunctions\\banque_donnees")
    matrice_centree, normes, numeros, positions, indice_pic_cible = charger_banque(DOSSIER)
    debut, fin = 0, 24000
    numeros_cases = (1, 5, 70)
    indices_points = tuple(numero - 1 for numero in numeros_cases)
    extraits = [matrice_centree[0, i, debut:fin] for i in indices_points]
    indices_pics = [premier_pic(signal) for signal in extraits]
    indice_cible = max(indices_pics)
    signaux_alignes = [
        np.pad(signal, (indice_cible - indice_pic, 0))[:len(signal)]
        for signal, indice_pic in zip(extraits, indices_pics)
    ]

    carte, corr = corr_function(
        matrice_centree[0, indices_points[0]], matrice_centree, debut, fin
    )
    for signal in signaux_alignes:
        plt.plot(signal)
    plt.axvline(indice_cible, color="black", linestyle="--")
    plt.show()
    
    print(carte.shape)
    plt.imshow(carte, cmap='coolwarm', vmin=-1, vmax=1)
    plt.show()
    