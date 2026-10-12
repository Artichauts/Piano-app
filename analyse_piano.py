import numpy as np
from pathlib import Path
from scipy.signal import resample_poly, butter, sosfiltfilt

BANQUE_DIR = Path(__file__).parent / "userfunctions" / "900p_banque_donnees"

def charger_donnees(banque_dir=BANQUE_DIR):
    """
    Charge les données de 891 points en un tableau de forme (3, 297, 24000)
    - 3 : nombre de réplicas
    - 297 : nombre de signaux
    - 24000 : nombre de échantillons par signal
    Les fichiers sont nommées 1.1.npy, 1.2.npy, 1.3.npy, ..., 297.1.npy, 297.2.npy, 297.3.npy
    """
    replicas = []
    for r in range(1, 4):  # 1, 2, 3
        replica_signals = []
        for i in range(1, 298):  # 1 to 297
            filename = f"{i}.{r}.npy"
            filepath = banque_dir / filename
            signal = np.load(filepath)
            replica_signals.append(signal)
        replicas.append(np.array(replica_signals))
    return np.array(replicas)  # forme (3, 297, 24000)

def degrader_bits(signaux, n_bits):
    """
    Dégrade la résolution en bits des signaux.
    signaux : de forme (3, 297, 24000)
    n_bits : int, nombre de bits à conserver (1 à 16)
    retourne signaux de même forme, mais avec des valeurs quantifiées.
    """
    niveaux = 2 ** (n_bits - 1)
    return np.round(signaux * niveaux) / niveaux

def degrader_fs(signaux, facteur):
    """
    Dégrade la fréquence d'échantillonnage des signaux par un facteur entier.
    signaux : de forme (3, 297, 24000)
    facteur : int, facteur de dégradation
    retourne signaux de forme (3, 297, 24000 // facteur)
    """
    return resample_poly(signaux, up=1, down=facteur, axis=-1)

def filtrer(signaux, fs, f_min, f_max):
    """
    Filtre les signaux selon les fréquences de coupure.
    signaux : de forme (3, 297, 24000)
    fs : float, fréquence d'échantillonnage
    f_min : float, fréquence de coupure basse
    f_max : float, fréquence de coupure haute
    retourne signaux filtrés de même forme.
    """
    if f_min is None:      # passe-bas : on garde tout ce qui est sous f_max
        sos = butter(4, f_max, btype="low", fs=fs, output="sos")
    else:                  # passe-bande : on garde entre f_min et f_max
        sos = butter(4, [f_min, f_max], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, signaux, axis=-1)

def ref_test(donnees, A_moy, B_moy, C_test):
    """
    Prend les données de forme (3, 297, x) et quels réplica devront être moyenner et lequel
    devra être celui de test. 
    A_moy : int; numéro du réplica à moyenner (1 ou 2 ou 3)
    B_moy : int, numéro du réplica à moyenner (1 ou 2 ou 3)
    C_test : int, numéro du réplica de test (1 ou 2 ou 3)
    Retourne un tableau de forme (297, x) pour le réplica de test et un tableau de forme 
    (297, x) pour la moyenne des deux autres réplica.
    """
    test = donnees[C_test - 1]  # C_test est 1-indexé
    ref = (donnees[A_moy - 1] + donnees[B_moy - 1]) / 2
    return ref, test # forme (297, x) pour les deux

# def corr_diag(ref, test):
#     """
#     Calcule la corrélation entre deux signaux de forme (297, 24000)
#     Retourne un tableau de forme (297,) avec les coefficients de corrélation pour chaque signal.
#     """
#     n_signals = ref.shape[0]
#     correlations = np.zeros(n_signals)
#     for i in range(n_signals):
#         correlations[i] = np.corrcoef(ref[i], test[i])[0, 1]
#     return correlations

def corr(ref, test):
    """Retourne C de forme (297, 297) : C[q, p] = corrélation entre le test q et la référence p."""
    ref_c  = ref  - ref.mean(axis=1, keepdims=True)    # on retire la moyenne de chaque signal
    test_c = test - test.mean(axis=1, keepdims=True)
    ref_n  = ref_c  / (np.linalg.norm(ref_c,  axis=1, keepdims=True) + 1e-12)   # on normalise
    test_n = test_c / (np.linalg.norm(test_c, axis=1, keepdims=True) + 1e-12)
    return test_n @ ref_n.T     # (297, x) @ (x, 297) -> (297, 297)

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

def evaluer_cnr(C):
    """CNR moyen sur les 297 points touchés (une ligne de C = un point touché)."""
    valeurs = np.array([cnr_2d(C[q]) if C[q].max() > 0 else np.nan
                        for q in range(C.shape[0])])
    return np.nanmean(valeurs)

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

def evaluer_fwhm(C):
    """FWHM moyen (cm) sur les 297 points touchés."""

    valeurs = np.array([fwhm_2d_area(C[q]) if C[q].max() > 0 else np.nan
                        for q in range(C.shape[0])])
    return np.nanmean(valeurs)

if __name__ == "__main__":
    BANQUE_DIR = Path(__file__).parent / "userfunctions" / "900p_banque_donnees"
    donnees = charger_donnees(BANQUE_DIR)
    #ref, test = ref_test(donnees, 2, 3, 1)
    #print(corr(ref, test).shape)  # devrait afficher un tableau de forme (297,)
    ref, test = ref_test(donnees, 1, 2, 3)
    C = corr(ref, test)
    print(C.shape)                                      # (297, 297)
    print("Diag mean",np.diag(C).mean())                            # devrait être élevé
    print("Accuracy",np.mean(np.argmax(C, axis=1) == np.arange(297)))  




# # --- 1. Données ---
# def charger_donnees(fichier):
#     """Retourne signaux de forme (N, M, n_repetitions, n_echantillons) et fs0."""

# # --- 2. Dégradations (une par graphique) ---
# def degrader_fs(signaux, fs0, facteur):      ...   # retourne signaux, nouvelle fs
# def degrader_bits(signaux, n_bits, pleine_echelle): ...
# def filtrer_bande(signaux, fs, f_min, f_max):    ...

# # --- 3. Carte de corrélation ---
# def tenseur_correlation(signaux_ref, signaux_test):
#     """Retourne corr_tensor de forme (N, M, N, M)."""

# # --- 4. Métriques (celles de ton coéquipier, légèrement adaptées) ---
# # cnr_2d, cnr_map_2d, fwhm_2d_area(map2d, taille_pixel), fwhm_map_2d

# # --- 5. Évaluer un niveau de dégradation ---
# def evaluer(signaux_ref, signaux_test, taille_pixel):
#     """Retourne (cnr_moyen, fwhm_moyen, erreur_loc_moyenne)."""

# # --- 6. Balayage + graphique ---
# def balayage(fonction_degrader, valeurs_param, signaux, ...):
#     """Pour chaque valeur : dégrader -> évaluer (sur les 3 rotations) -> moyenne et écart-type."""

# def tracer(valeurs_param, resultats, xlabel): ...  # CNR et FWHM sur deux sous-graphiques