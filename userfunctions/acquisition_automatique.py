
import time 
import queue
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

# ---------- Paramètres ----------
FS = 48000               # Fréquence d'échantillonnage [Hz] (doit correspondre à votre acquisition)
BLOCK = 256              # Taille des blocs audio
SEUIL = 0.15             # Seuil de déclenchement (signal normalisé entre -1 et 1)
T_AVANT = 0.1            # Durée gardée avant le déclenchement [s]
T_APRES = 0.4            # Durée gardée après le déclenchement [s]
DOSSIER = Path(__file__).resolve().parent / "banque_donnees"
APERCU = False           # Afficher le signal avant de le nommer
N_POINTS = 297
N_REPLIQUES = 3


def creer_flux(q, fs=FS, block=BLOCK):
    """Ouvre le micro en continu; chaque bloc audio est déposé dans la file q."""
    def callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(indata[:, 0].copy())

    return sd.InputStream(samplerate=fs, channels=1, blocksize=block,
                          dtype="float32", callback=callback)


def vider(q):
    """Jette tout ce qui s'est accumulé dans la file."""
    while not q.empty():
        q.get_nowait()


def attendre_impact(q, seuil=SEUIL, t_avant=T_AVANT, t_apres=T_APRES, fs=FS, block=BLOCK):
    """Attend un impact et retourne un signal de longueur fixe :
    t_avant secondes avant le déclenchement + t_apres secondes après."""
    n_avant = int(t_avant * fs)
    n_apres = int(t_apres * fs)

    # Tampon circulaire : garde juste assez de blocs pour couvrir n_avant
    anneau = deque(maxlen=int(np.ceil(n_avant / block)) + 1)

    while True:
        bloc = q.get()
        anneau.append(bloc)
        if len(anneau) < anneau.maxlen:     # tampon pas encore plein
            continue
        depasse = np.abs(bloc) > seuil
        if depasse.any():
            print("Déclenchement détecté !")
            break

    # Position du déclenchement dans les données concaténées
    donnees = np.concatenate(anneau)
    i_trig = len(donnees) - len(bloc) + np.argmax(depasse)

    # On continue d'enregistrer jusqu'à avoir assez d'échantillons après
    blocs = [donnees]
    n_total = len(donnees)
    while n_total < i_trig + n_apres:
        b = q.get()
        blocs.append(b)
        n_total += len(b)
    donnees = np.concatenate(blocs)

    return donnees[i_trig - n_avant : i_trig + n_apres]


def apercu_signal(signal, t_avant=T_AVANT, fs=FS):
    """Trace le signal (t = 0 au déclenchement). Fermer la fenêtre pour continuer."""
    t = (np.arange(len(signal)) / fs - t_avant) * 1000
    plt.plot(t, signal)
    plt.axvline(0, color="r", ls="--", lw=0.8)
    plt.xlabel("Temps [ms]")
    plt.ylabel("Amplitude")
    plt.show()


def sauvegarder(signal, nom, dossier=DOSSIER):
    """Sauvegarde le signal en .npy dans le dossier de la banque."""
    dossier.mkdir(parents=True, exist_ok=True)
    chemin = dossier / f"{nom}.npy"
    np.save(chemin, signal)
    return chemin


def acquisition_banque():
    """Enregistre automatiquement trois impacts pour chacun des 297 points."""
    q = queue.Queue()
    with creer_flux(q):
        for point in range(1, N_POINTS + 1):
            for replique in range(1, N_REPLIQUES + 1):
                nom = f"{point}.{replique}"
                if (DOSSIER / f"{nom}.npy").exists():
                    print(f"Déjà présent, conservé : {nom}.npy")
                    continue
                time.sleep(0.4)
                print("attendre avant acquisition...")
                vider(q)
                print(f"Point {point}/{N_POINTS}, réplique {replique}/{N_REPLIQUES} : en attente d'un impact...")
                signal = attendre_impact(q)
                print("Impact enregistré !")

                if APERCU:
                    apercu_signal(signal)

                print("Sauvegardé :", sauvegarder(signal, nom, DOSSIER))

    print("Acquisition terminée : les 297 points ont été traités.")


if __name__ == "__main__":
    try:
        acquisition_banque()
    except KeyboardInterrupt:
        print("\nArrêt.")