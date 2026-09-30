import numpy as np
from pathlib import Path

N_LIGNES = 11
N_COLONNES = 27  # 297 / 11

plage = {
    "do": [(1, 38)],
    "do#": [(39, 44), (50, 55), (61, 66)],
    "ré": [(45, 49), (56, 60), (67, 77), (78, 82)],
    "ré#": [(83, 88), (94, 99)],
    "mi": [(89, 93), (100, 132)],
    "fa": [(133, 159)],
    "fa#": [(160, 165), (171, 176), (182, 187)],
    "sol": [(166, 170), (177, 181), (188, 203)],
    "sol#": [(204, 209), (215, 220), (226, 231)],
    "la": [(210, 214), (221, 225), (232, 247)],
    "la#": [(248, 253), (259, 264), (270, 275)],
    "si": [(254, 258), (265, 269), (276, 297)],
    }

def case_vers_position(k):
    """Numéro de case (1 à 297) -> (ligne, colonne), 0-indexé, ligne 0 = en bas."""
    return (k - 1) % N_LIGNES, (k - 1) // N_LIGNES


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
            longueur = min(len(signal) for signal in repliques)
            signal = np.mean([rep[:longueur] for rep in repliques], axis=0)
        elif 0 in fichiers:
            signal = np.load(fichiers[0])
        else:
            continue
        numeros.append(numero)
        signaux.append(signal)

    if not signaux:
        raise ValueError("Aucun point complet trouvé dans la banque de données.")

    n = min(len(s) for s in signaux)
    matrice = np.stack([s[:n] for s in signaux])
    matrice_centree = matrice - matrice.mean(axis=1, keepdims=True)
    normes = np.linalg.norm(matrice_centree, axis=1)

    positions = np.array([case_vers_position(k) for k in numeros])
    return matrice_centree, normes, numeros, positions


def analyser_signal(signal, matrice_centree, normes, numeros, positions, table_notes):
    """Retourne la note, la carte de corrélation 2D, et le vecteur brut de corrélations."""
    signal = np.asarray(signal, dtype=float)
    if signal.ndim > 1:
        signal = signal[:, 0]

    n = matrice_centree.shape[1]
    if len(signal) != n:
        signal = signal[:n] if len(signal) > n else np.pad(signal, (0, n - len(signal)))

    s = signal - signal.mean()
    corr = (matrice_centree @ s) / (normes * np.linalg.norm(s) + 1e-12)

    carte_corr = np.full((N_LIGNES, N_COLONNES), np.nan)
    carte_corr[positions[:, 0], positions[:, 1]] = corr

    i_best = int(np.argmax(corr))
    note = table_notes.get(numeros[i_best])

    return note, carte_corr, corr

