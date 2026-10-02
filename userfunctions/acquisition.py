import queue
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

# ---------- Paramètres ----------
FS = 48000               # Fréquence d'échantillonnage [Hz] (doit correspondre à votre acquisition)
BLOCK = 256              # Taille des blocs audio
SEUIL = 0.060             # Seuil de déclenchement (signal normalisé entre -1 et 1)
T_AVANT = 0.0            # Durée gardée avant le déclenchement [s]
T_APRES = 0.1            # Durée gardée après le déclenchement [s]
DOSSIER = Path("Piano\\Piano-app\\userfunctions\\banque_donnees")
APERCU = False            # Afficher le signal avant de le nommer


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
        if len(anneau) < anneau.maxlen:      # tampon pas encore plein
            continue
        depasse = np.abs(bloc) > seuil
        if depasse.any():
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
    dossier.mkdir(exist_ok=True)
    chemin = dossier / f"{nom}.npy"
    np.save(chemin, signal)
    return chemin


def acquisition_banque():
    """Boucle principale : impact -> aperçu -> nom -> sauvegarde."""
    q = queue.Queue()
    with creer_flux(q):
        while True:
            vider(q)   # ignore ce qui s'est passé pendant qu'on nommait/regardait
            print("\nEn attente d'un impact...")
            signal = attendre_impact(q)
            print("Impact enregistré !")

            if APERCU:
                apercu_signal(signal)

            nom = input("Nom du fichier (vide = rejeter, q = quitter) : ").strip()
            if nom.lower() == "q":
                break
            if not nom:
                print("Signal rejeté.")
                continue
            if (DOSSIER / f"{nom}.npy").exists():
                if input("Ce fichier existe déjà, écraser ? (o/n) : ").lower() != "o":
                    print("Signal rejeté.")
                    continue
            print("Sauvegardé :", sauvegarder(signal, nom))


if __name__ == "__main__":
    try:
        acquisition_banque()
    except KeyboardInterrupt:
        print("\nArrêt.")