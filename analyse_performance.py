"""
Mandat 2 (PHS3910) - Performances du piano acoustique

Résolution (FWHM) et contraste (CNR) en fonction de :
  1. la profondeur de bits
  2. la fréquence d'échantillonnage
  3. le contenu fréquentiel

Chaîne de traitement :
  charger -> aligner -> dégrader -> (référence, test) -> corrélation -> métriques -> graphique
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import resample_poly, butter, sosfiltfilt, correlate, correlation_lags

# =============================================================================
# PARAMÈTRES
# =============================================================================
BANQUE_DIR = Path(__file__).parent / "userfunctions" / "900p_banque_donnees"

FS0 = 48000                      # fréquence d'échantillonnage d'origine (Hz)
N_LIGNES, N_COLONNES = 11, 27    # grille de la plaque
TAILLE_PIXEL = 1.0               # une case = 1 cm

# (réplica à moyenner, réplica à moyenner, réplica de test)
ESSAIS = [(2, 3, 1), (1, 3, 2), (1, 2, 3)]

ALIGNER = True                   # mettre False pour comparer avec / sans alignement

VALEURS_BITS = [16, 14, 12, 10, 8, 7, 6, 5, 4]  # profondeur de bits
FACTEURS_FS = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48]      # fs = 48000 / facteur
COUPURES = [250, 500, 750, 1000, 1500, 2000, 3000]     # passe-bas (Hz)


# =============================================================================
# 1. CHARGEMENT
# =============================================================================
def charger_donnees(banque_dir=BANQUE_DIR):
    """
    Charge les données de 891 points en un tableau de forme (3, 297, 24000)
    - 3 : nombre de réplicas
    - 297 : nombre de signaux
    - 24000 : nombre d'échantillons par signal
    Les fichiers sont nommés 1.1.npy, 1.2.npy, 1.3.npy, ..., 297.1.npy, 297.2.npy, 297.3.npy
    """
    replicas = []
    for r in range(1, 4):  # 1, 2, 3
        replica_signals = []
        for i in range(1, 298):  # 1 à 297
            filepath = banque_dir / f"{i}.{r}.npy"
            replica_signals.append(np.load(filepath))
        replicas.append(np.array(replica_signals))
    return np.array(replicas)  # forme (3, 297, 24000)


# =============================================================================
# 2. ALIGNEMENT SUR LE PREMIER PIC (repris du code temps réel)
# =============================================================================
def premier_pic(signal, seuil_relatif=0.06):
    """Indice du premier maximum local dont l'amplitude dépasse 6 % du maximum."""
    amplitude = np.abs(signal - signal[0])
    seuil = amplitude.max() * seuil_relatif
    pics = np.flatnonzero(
        (amplitude[1:-1] >= amplitude[:-2])
        & (amplitude[1:-1] > amplitude[2:])
        & (amplitude[1:-1] >= seuil)
    ) + 1
    return int(pics[0]) if pics.size else int(np.argmax(amplitude))


def aligner_matrice_sur_premier_pic(matrice):
    """
    Décale chaque signal pour que tous aient leur premier pic au même indice
    (le plus tardif de tous). Fonctionne sur la forme (3, 297, 24000).
    Retourne (matrice alignée, indice cible).
    """
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


def decalages_entre_replicas(donnees, debut=4000, fin=10000, lag_max=300):
    """
    DIAGNOSTIC seulement : pour chaque point, décalage (en échantillons) qui fait le mieux
    ressembler le réplica 2 au réplica 1. Plus c'est proche de 0, mieux c'est aligné.
    (On ne l'utilise pas pour aligner, car il utilise la connaissance du bon point.)
    """
    n = fin - debut
    lags = correlation_lags(n, n)
    garder = np.abs(lags) <= lag_max
    lags = lags[garder]
    resultat = np.zeros(donnees.shape[1], dtype=int)
    for q in range(donnees.shape[1]):
        a = donnees[0, q, debut:fin].astype(float)
        b = donnees[1, q, debut:fin].astype(float)
        a -= a.mean()
        b -= b.mean()
        c = correlate(a, b, mode="full", method="fft")[garder]
        resultat[q] = lags[np.argmax(c)]
    return resultat


# =============================================================================
# 3. DÉGRADATIONS
# =============================================================================
def degrader_bits(signaux, n_bits):
    """
    Dégrade la résolution en bits des signaux (pleine échelle = +/- 1).
    n_bits : int, entre 4 et 16 (en dessous de 4, le signal devient nul)
    Retourne des signaux de même forme, avec des valeurs quantifiées.
    """
    niveaux = 2 ** (n_bits - 1)
    return np.round(signaux * niveaux) / niveaux


def degrader_fs(signaux, facteur):
    """
    Dégrade la fréquence d'échantillonnage par un facteur entier (avec filtre anti-repliement).
    Retourne des signaux de forme (..., 24000 // facteur).
    """
    return resample_poly(signaux, up=1, down=facteur, axis=-1)


def filtrer(signaux, fs, f_min, f_max):
    """
    Filtre de Butterworth d'ordre 4 (sans décalage dans le temps).
    f_min = None : passe-bas (on garde ce qui est sous f_max)
    sinon        : passe-bande entre f_min et f_max
    """
    if f_min is None:
        sos = butter(4, f_max, btype="low", fs=fs, output="sos")
    else:
        sos = butter(4, [f_min, f_max], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, signaux, axis=-1)


# =============================================================================
# 4. RÉFÉRENCE / TEST ET CORRÉLATION
# =============================================================================
def ref_test(donnees, A_moy, B_moy, C_test):
    """
    A_moy, B_moy : numéros (1, 2 ou 3) des réplicas à moyenner pour la référence
    C_test       : numéro du réplica de test
    Retourne (ref, test), tous deux de forme (297, x).
    """
    test = donnees[C_test - 1]
    ref = (donnees[A_moy - 1] + donnees[B_moy - 1]) / 2
    return ref, test


def corr(ref, test):
    """Retourne C de forme (297, 297) : C[q, p] = corrélation entre le test q et la référence p."""
    ref_c = ref - ref.mean(axis=1, keepdims=True)      # on retire la moyenne de chaque signal
    test_c = test - test.mean(axis=1, keepdims=True)
    ref_n = ref_c / (np.linalg.norm(ref_c, axis=1, keepdims=True) + 1e-12)     # on normalise
    test_n = test_c / (np.linalg.norm(test_c, axis=1, keepdims=True) + 1e-12)
    return test_n @ ref_n.T     # (297, x) @ (x, 297) -> (297, 297)


# =============================================================================
# 5. MÉTRIQUES 
# =============================================================================
def cnr_2d(map2d):
    """
    Calcule le CNR (Contrast-to-Noise Ratio) d'une carte de corrélation.
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

    return (mean_signal - mean_background) / std_background


def fwhm_2d_area(map2d):
    """
    Calcule un FWHM basé sur l'aire de la région >= half_max,
    converti en diamètre équivalent (aire d'un disque) en cm.
    """
    max_val = np.max(map2d)
    half_max = max_val / 2.0

    area = np.sum(map2d >= half_max)  # nombre de cases au-dessus de half_max
    if area == 0:
        return 0.0

    # Diamètre équivalent d'un disque de même aire : A = pi*(d/2)^2
    return 2 * np.sqrt(area / np.pi) * TAILLE_PIXEL


def evaluer_cnr(C):
    """CNR moyen sur les 297 points touchés (une ligne de C = un point touché)."""
    valeurs = np.array([cnr_2d(C[q]) if C[q].max() > 0 else np.nan
                        for q in range(C.shape[0])])
    return np.nanmean(valeurs)


def evaluer_fwhm(C):
    """FWHM moyen (cm) sur les 297 points touchés."""
    valeurs = np.array([fwhm_2d_area(C[q]) if C[q].max() > 0 else np.nan
                        for q in range(C.shape[0])])
    return np.nanmean(valeurs)


def evaluer_donnees(donnees):
    """Pour des données (3, 297, x), retourne les CNR et FWHM des 3 essais (3 valeurs chacun)."""
    cnr, fwhm = [], []
    for a, b, c in ESSAIS:
        ref, test = ref_test(donnees, a, b, c)
        C = corr(ref, test)
        cnr.append(evaluer_cnr(C))
        fwhm.append(evaluer_fwhm(C))
    return np.array(cnr), np.array(fwhm)


# =============================================================================
# 6. VÉRIFICATION (sans dégradation)
# =============================================================================
def erreur_localisation(C):
    """
    Distance médiane (cm) entre le vrai point touché et le point où la corrélation est maximale.
    Numérotation : par colonne, de bas en haut (point k -> ligne k % 11, colonne k // 11).
    """
    vrai = np.arange(C.shape[0])
    trouve = np.argmax(C, axis=1)
    d_ligne = vrai % N_LIGNES - trouve % N_LIGNES
    d_col = vrai // N_LIGNES - trouve // N_LIGNES
    return np.median(np.hypot(d_ligne, d_col)) * TAILLE_PIXEL


def bilan(C):
    print("Corrélation moyenne (même point)      :", round(float(np.diag(C).mean()), 3))
    print("Exactitude (bonne case exacte)        :", round(float(np.mean(np.argmax(C, axis=1) == np.arange(C.shape[0]))), 3))
    print("Erreur de localisation médiane (cm)   :", round(float(erreur_localisation(C)), 2))
    lags = decalages_entre_replicas(donnees)             # donnees = alignées
    print(np.percentile(np.abs(lags), [50, 90, 99, 100]))
    print("points mal alignés (> 20 éch.) :", np.sum(np.abs(lags) > 20))


# =============================================================================
# 7. BALAYAGE ET GRAPHIQUES
# =============================================================================
def balayage(donnees, degrader, valeurs):
    """
    Pour chaque valeur du paramètre : dégrade les données, puis évalue les 3 essais.
    degrader : fonction (donnees, valeur) -> donnees dégradées
    Retourne (cnr_moyen, cnr_ecart_type, fwhm_moyen, fwhm_ecart_type), un nombre par valeur.
    """
    cnr_moy, cnr_std, fwhm_moy, fwhm_std = [], [], [], []
    for v in valeurs:
        deg = degrader(donnees, v)
        print(f"  signaux entièrement nuls : {100*np.mean(np.all(deg == 0, axis=-1)):.1f} %")
        cnr, fwhm = evaluer_donnees(deg)
        cnr_moy.append(cnr.mean())
        cnr_std.append(cnr.std())
        fwhm_moy.append(fwhm.mean())
        fwhm_std.append(fwhm.std())
        print(
            f"  valeur = {v} : "
            f"CNR = {cnr.mean():.2f} ± {cnr.std():.2f}, "
            f"FWHM = {fwhm.mean():.2f} ± {fwhm.std():.2f} cm"
        )
    return np.array(cnr_moy), np.array(cnr_std), np.array(fwhm_moy), np.array(fwhm_std)


def tracer(x, resultats, xlabel, fichier_cnr, fichier_fwhm, log_x=False):
    """Trace et enregistre séparément le CNR et le FWHM, avec barres d'erreur."""
    cnr_moy, cnr_std, fwhm_moy, fwhm_std = resultats
    fig_cnr, ax_cnr = plt.subplots(figsize=(7, 4))
    ax_cnr.errorbar(x, cnr_moy, yerr=cnr_std, marker="o", capsize=3)
    #ax_cnr.set_title("Contraste (CNR)")
    ax_cnr.set_ylabel("CNR")
    ax_cnr.set_xlabel(xlabel)
    ax_cnr.grid(True, alpha=0.3)

    fig_fwhm, ax_fwhm = plt.subplots(figsize=(7, 4))
    ax_fwhm.errorbar(x, fwhm_moy, yerr=fwhm_std, marker="o", capsize=3)
    #ax_fwhm.set_title("Résolution (FWHM)")
    ax_fwhm.set_ylabel("FWHM (cm)")
    ax_fwhm.set_xlabel(xlabel)
    ax_fwhm.grid(True, alpha=0.3)

    if log_x:
        ax_cnr.set_xscale("log")
        ax_fwhm.set_xscale("log")

    fig_cnr.tight_layout()
    fig_fwhm.tight_layout()
    dossier_sortie = Path(__file__).parent
    fig_cnr.savefig(dossier_sortie / fichier_cnr, dpi=150)
    fig_fwhm.savefig(dossier_sortie / fichier_fwhm, dpi=150)
    return fig_cnr, fig_fwhm

# =============================================================================
# 8. FFT moyen
# =============================================================================

def spectre_moyen(donnees, fs=FS0):
    """Spectre de puissance moyen (FFT) sur tous les signaux. Retourne (fréquences en Hz, puissance)."""
    S = np.zeros(donnees.shape[-1] // 2 + 1)
    for replique in donnees:                       # un réplica à la fois pour économiser la mémoire
        S += (np.abs(np.fft.rfft(replique, axis=-1)) ** 2).mean(axis=0)
    S /= donnees.shape[0]
    return np.fft.rfftfreq(donnees.shape[-1], d=1 / fs), S

def spectre_moyen_normalise(donnees, fs=FS0):
    """Comme spectre_moyen, mais chaque signal est normalisé par son énergie."""
    S = np.zeros(donnees.shape[-1] // 2 + 1)
    for replique in donnees:
        P = np.abs(np.fft.rfft(replique, axis=-1)) ** 2
        P /= P.sum(axis=1, keepdims=True) + 1e-30
        S += P.mean(axis=0)
    return np.fft.rfftfreq(donnees.shape[-1], d=1 / fs), S / donnees.shape[0]

def tracer_spectre(donnees, fichier="spectre-moyen.png", f_max=5000):
    """Trace le spectre moyen, l'enregistre et affiche la fréquence du pic et celles de 90 % et 99 % de l'énergie."""
    freqs, S = spectre_moyen_normalise(donnees)
    cumul = np.cumsum(S) / S.sum()
    f_pic = freqs[np.argmax(S[1:]) + 1]            # on ignore la composante à 0 Hz
    f90 = freqs[np.searchsorted(cumul, 0.90)]
    f99 = freqs[np.searchsorted(cumul, 0.99)]
    print(f"Pic principal : {f_pic:.0f} Hz | 90 % de l'énergie sous {f90:.0f} Hz | 99 % sous {f99:.0f} Hz")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.semilogy(freqs, S / S.max())
    ax.axvline(f99, color="gray", linestyle="--", label=f"99 % de l'énergie : {f99:.0f} Hz")
    ax.set_xlim(0, f_max)
    ax.set_ylim(1e-6, 2)
    ax.set_xlabel("Fréquence (Hz)")
    ax.set_ylabel("Puissance relative")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / fichier, dpi=200)
    return f_pic, f90, f99

if __name__ == "__main__":
    donnees = charger_donnees(BANQUE_DIR)

    if ALIGNER:
        donnees, indice_cible = aligner_matrice_sur_premier_pic(donnees)
        print("Indice cible d'alignement :", indice_cible)

    # --- Vérification sans dégradation (à regarder avant de lancer les balayages)
    print("\n--- Sans dégradation (réplicas 1 et 2 en référence, 3 en test) ---")
    ref, test = ref_test(donnees, 1, 2, 3)
    bilan(corr(ref, test))
    lags = decalages_entre_replicas(donnees)
    print("Décalage médian entre réplicas (échantillons) :", np.median(np.abs(lags)))

    # --- 1. Profondeur de bits
    print("\n--- Profondeur de bits ---")
    res_bits = balayage(donnees, degrader_bits, VALEURS_BITS)
    tracer(VALEURS_BITS, res_bits, "Profondeur de bits",
           "cnr_bit.png", "FWHM_bit.png")

    # --- 2. Fréquence d'échantillonnage
    print("\n--- Fréquence d'échantillonnage ---")
    res_fs = balayage(donnees, degrader_fs, FACTEURS_FS)
    tracer(FS0 / np.array(FACTEURS_FS), res_fs, "Fréquence d'échantillonnage (Hz)",
           "cnr_fs.png", "FWHM_fs.png", log_x=True)

    # --- 3. Contenu fréquentiel (passe-bas)
    # "lambda d, f: ..." est une mini-fonction : elle prend les données d et la valeur f,
    # et appelle filtrer avec les bons arguments.
    print("\n--- Contenu fréquentiel (passe-bas) ---")
    res_filtre = balayage(donnees, lambda d, f: filtrer(d, FS0, None, f), COUPURES)
    tracer(COUPURES, res_filtre, "Fréquence de coupure du passe-bas (Hz)",
           "cnr_filtre.png", "FWHM_filtre.png")


    C = corr(*ref_test(donnees, 1, 2, 3)); q = 148        # un point au milieu
    plt.imshow(C[q].reshape(27, 11).T, origin="lower", cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(); plt.show(); print("fond médian :", np.median(C[q]))

    tracer_spectre(donnees)

    # Variante possible avec des bandes (f_min, f_max) :
    # BANDES = [(0.1, 500), (500, 1000), (1000, 2000), (2000, 4000)]
    # res_bandes = balayage(donnees, lambda d, b: filtrer(d, FS0, b[0], b[1]), BANDES)

    #plt.show()
