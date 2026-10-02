from collections import deque
import numpy as np
import pyqtgraph as pg
import sounddevice as sd

import time

from pathlib import Path
from PyQt6.QtMultimedia import QSoundEffect
from PyQt6.QtCore import QThread, pyqtSignal, QObject, QUrl
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QPushButton, QButtonGroup, QLabel)

from userfunctions.analyse_sig import (
    charger_banque,
    analyser_signal,
    aligner_signal_sur_pic,
    construire_table_notes,
)

pg.setConfigOptions(imageAxisOrder='row-major')

# ---------- Paramètres ----------
T = 0.1
FS = 48000
THRESHOLD = 0.06

T_AVANT, T_APRES = 0, 0.05
REFRACTAIRE = 0.15
CORR_DEBUT = 0
CORR_FIN = int(T_APRES * FS)

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
table_notes = construire_table_notes(plage)

# ---------- Conversion note française -> nom de fichier anglais ----------
LETTRE = {
    "do": "c", "do#": "c-", "ré": "d", "ré#": "d-",
    "mi": "e", "fa": "f", "fa#": "f-",
    "sol": "g", "sol#": "g-", "la": "a", "la#": "a-", "si": "b",
}
OCTAVES = {"Grave": 3, "Moyen": 4, "Aigu": 5}

# ---------- Chargement de la banque (une seule fois, au démarrage) ----------
BANQUE_DIR = Path(__file__).parent / "userfunctions" / "banque_donnees"
matrice_centree, normes, numeros, positions, indice_pic_cible = charger_banque(BANQUE_DIR)
matrice_centree = matrice_centree[:, :, CORR_DEBUT:CORR_FIN]
matrice_centree -= matrice_centree.mean(axis=2, keepdims=True)
normes = np.linalg.norm(matrice_centree, axis=2)

NOTES_DIR = Path(__file__).parent / "Wav-Notes"
NOTE_FILES = {f.stem: f for f in NOTES_DIR.glob("*.wav")}   # {"c3": Path, "c-3": Path, ...}
COOLDOWN = 1.0


# ---------- Détection de l'impact dans le flux (précision à l'échantillon près) ----------

class DetecteurImpact:
    def __init__(self, seuil, t_avant, t_apres, fs, refractaire=REFRACTAIRE):
        self.seuil = seuil
        self.fs = fs
        self.n_avant = int(t_avant * fs)
        self.n_apres = int(t_apres * fs)
        self.n_refractaire = int(refractaire * fs)
        self.tampon = np.zeros(0)
        self.i_trig = None
        self.silence_restant = 0
        self.temps_impact = None

    def ajouter_bloc(self, bloc):
        bloc = np.asarray(bloc).reshape(-1)

        if self.silence_restant > 0:
            self.silence_restant -= len(bloc)
            return None

        self.tampon = np.concatenate([self.tampon, bloc])

        if self.i_trig is None:
            depasse = np.where(np.abs(self.tampon) > self.seuil)[0]
            if depasse.size == 0:
                if len(self.tampon) > self.n_avant:
                    self.tampon = self.tampon[-self.n_avant:]
                return None
            self.i_trig = depasse[0]
            maintenant = time.monotonic()
            echantillons_apres_impact = len(self.tampon) - 1 - self.i_trig
            self.temps_impact = maintenant - echantillons_apres_impact / self.fs

        if len(self.tampon) >= self.i_trig + self.n_apres:
            debut = max(0, self.i_trig - self.n_avant)
            fenetre = self.tampon[debut: self.i_trig + self.n_apres]
            self.tampon = np.zeros(0)
            self.i_trig = None
            self.silence_restant = self.n_refractaire
            temps_impact = self.temps_impact
            self.temps_impact = None
            return fenetre, temps_impact

        return None


class SDRWorker(QObject):
    result_ready = pyqtSignal(object, object, object, object)

    def __init__(self, T=T, fs=FS):
        super().__init__()
        self.T, self.fs = T, fs
        self.detecteur = DetecteurImpact(THRESHOLD, T_AVANT, T_APRES, fs)
        self.running = True

    def run(self):
        n_bloc = int(self.T * self.fs)
        with sd.InputStream(samplerate=self.fs, channels=1, dtype="float32") as stream:
            while self.running:
                bloc, overflowed = stream.read(n_bloc)
                if overflowed:
                    print("Débordement du flux audio.")
                bloc = np.asarray(bloc)
                capture = self.detecteur.ajouter_bloc(bloc)

                carte_corr, note = None, None
                temps_impact = None
                if capture is not None:
                    fenetre, temps_impact = capture
                    fenetre = aligner_signal_sur_pic(fenetre, indice_pic_cible)
                    fenetre = fenetre[CORR_DEBUT:CORR_FIN]
                    note, carte_corr, _ = analyser_signal(
                        fenetre, matrice_centree, normes, numeros, positions, table_notes)

                self.result_ready.emit(bloc, carte_corr, note, temps_impact)

    def stop(self):
        self.running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._closing = False
        self.octave = OCTAVES["Moyen"]   # 4 par défaut

        # ---- Signal plot ----
        self.plot = pg.PlotWidget()
        self.plot.setLabel('bottom', 'Time', units='s')
        self.plot.setLabel('left', 'Amplitude')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setDownsampling(auto=True, mode='peak')
        self.plot.setClipToView(True)
        self.curve = self.plot.plot(pen=pg.mkPen(width=1))

        for y in (THRESHOLD, -THRESHOLD):
            self.plot.addItem(pg.InfiniteLine(pos=y, angle=0, movable=False,
                                              pen=pg.mkPen('r', style=pg.QtCore.Qt.PenStyle.DashLine)))

        # ---- Correlation map plot ----
        self.corr_plot = pg.PlotWidget(title="Carte de corrélation")
        self.corr_img = pg.ImageItem()

        cmap = pg.colormap.get('viridis')
        self.corr_img.setColorMap(cmap)

        self.colorbar = pg.ColorBarItem(colorMap=cmap, label="Coefficient de corrélation")
        self.colorbar.setImageItem(self.corr_img, insert_in=self.corr_plot.getPlotItem())

        self.corr_plot.addItem(self.corr_img)
        self.corr_plot.setAspectLocked(True)

        # ---- Layout ----
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self._creer_selecteur_octave())
        layout.addWidget(self.plot, stretch=1)
        layout.addWidget(self.corr_plot, stretch=2)
        self.setCentralWidget(central)

        # ---- Sound ----
        self.sound = QSoundEffect()
        self.sound.setVolume(1.0)
        self.sound.playingChanged.connect(self._mesurer_latence_audio)
        self.temps_impact_en_attente = None
        self.last_play = 0.0

        # ---- Worker thread ----
        self.sdr_thread = QThread()
        self.worker = SDRWorker()
        self.worker.moveToThread(self.sdr_thread)

        self.worker.result_ready.connect(self.on_result)

        self.sdr_thread.started.connect(self.worker.run)
        self.sdr_thread.start()

    def _creer_selecteur_octave(self):
        """Petit sélecteur grave/moyen/aigu, aligné à droite."""
        conteneur = QWidget()
        ligne = QHBoxLayout(conteneur)
        ligne.setContentsMargins(0, 0, 0, 0)
        self.latence_label = QLabel("Délai : -- ms")
        self.latence_label.setMinimumWidth(125)
        ligne.addWidget(self.latence_label)
        ligne.addStretch()
        ligne.addWidget(QLabel("Octave :"))

        self.groupe_octave = QButtonGroup(self)
        self.groupe_octave.setExclusive(True)
        for nom, octave in OCTAVES.items():
            bouton = QPushButton(nom)
            bouton.setCheckable(True)
            bouton.clicked.connect(lambda _, o=octave: setattr(self, "octave", o))
            ligne.addWidget(bouton)
            self.groupe_octave.addButton(bouton)
            if octave == self.octave:
                bouton.setChecked(True)

        return conteneur

    def jouer_note(self, note, temps_impact=None):
        if note is None or time.monotonic() - self.last_play < COOLDOWN:
            return None
        lettre = LETTRE.get(note)
        if lettre is None:
            print(f"Note inconnue : {note}")
            return None
        stem = f"{lettre}{self.octave}"
        fichier = NOTE_FILES.get(stem)
        if fichier is None:
            print(f"Aucun fichier son pour : {stem}")
            return None
        self.sound.setSource(QUrl.fromLocalFile(str(fichier.resolve())))
        self.temps_impact_en_attente = temps_impact
        self.sound.play()
        self.last_play = time.monotonic()
        return stem

    def _mesurer_latence_audio(self):
        if self._closing:
            return
        if self.sound.isPlaying() and self.temps_impact_en_attente is not None:
            latence_ms = (time.monotonic() - self.temps_impact_en_attente) * 1000
            self.latence_label.setText(f"Délai : {latence_ms:.1f} ms")
            print(f"Latence appui -> démarrage du son : {latence_ms:.1f} ms")
            self.temps_impact_en_attente = None

    def on_result(self, bloc, carte_corr, note, temps_impact):
        if self._closing:
            return
        data = bloc[:, 0] if bloc.ndim > 1 else bloc
        t = np.arange(len(data)) / FS
        self.curve.setData(t, data)

        if carte_corr is not None:
            self.corr_img.setImage(np.asarray(carte_corr), autoLevels=True)
            joue = self.jouer_note(note, temps_impact)
            titre = f"Carte de corrélation, note : {note}"
            if joue:
                titre += f" | jouée : {joue}"
            self.corr_plot.setTitle(titre)

    def closeEvent(self, event):
        self._closing = True
        self.worker.result_ready.disconnect(self.on_result)
        self.sound.playingChanged.disconnect(self._mesurer_latence_audio)
        self.sound.stop()
        self.temps_impact_en_attente = None
        self.worker.stop()
        self.sdr_thread.quit()
        self.sdr_thread.wait()
        super().closeEvent(event)


app = QApplication([])
window = MainWindow()
window.show()
app.exec()