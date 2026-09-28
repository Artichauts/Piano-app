from collections import deque
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QThread, pyqtSignal, QObject, QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout

from userfunctions.RecordMicro import record_audio
from userfunctions.analyse_signal import analyse_signal 

T, FS = 0.1, 48000
THRESHOLD = 0.05 


N_SAMPLES = int(round(3 * T * FS))      # window = 3 chunks -> 14400 samples
GRID = (20, 20)                          # (ny, nx) of touch positions
N_POS = GRID[0] * GRID[1]                # 400 positions

rng = np.random.default_rng(0)
matrix = rng.standard_normal((N_POS, N_SAMPLES))  

class SDRWorker(QObject):
    end_of_run = pyqtSignal()
    result_ready = pyqtSignal(object, object, object)   # window, corr_map, note

    def __init__(self, T=T, fs=FS):
        super().__init__()
        self.T, self.fs = T, fs
        self.chunks = deque(maxlen=3)   # [previous, current, next]

    def run(self):
        new = np.asarray(record_audio(self.T, self.fs))
        self.chunks.append(new)

        if len(self.chunks) == 3:
            window = np.concatenate(self.chunks, axis=0)

            # Peak amplitude of the "current" chunk (middle one)
            current = self.chunks[1]
            peak = np.max(np.abs(current))

            corr_map, note = None, None
            if peak > THRESHOLD:
                corr_map, note = analyse_signal(window, matrix)

            self.result_ready.emit(window, corr_map, note)

        self.end_of_run.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # ---- Signal plot ----
        self.plot = pg.PlotWidget()
        self.plot.setLabel('bottom', 'Time', units='s')
        self.plot.setLabel('left', 'Amplitude')
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setXRange(-T, 2 * T, padding=0)
        self.plot.setDownsampling(auto=True, mode='peak')
        self.plot.setClipToView(True)
        self.curve = self.plot.plot(pen=pg.mkPen(width=1))

        self.region = pg.LinearRegionItem(values=(0, T), movable=False,
                                          brush=pg.mkBrush(100, 150, 255, 40))
        self.plot.addItem(self.region)

        # Threshold lines (+/-)
        for y in (THRESHOLD, -THRESHOLD):
            self.plot.addItem(pg.InfiniteLine(pos=y, angle=0, movable=False,
                                              pen=pg.mkPen('r', style=pg.QtCore.Qt.PenStyle.DashLine)))

        # ---- Correlation map plot ----
        self.corr_plot = pg.PlotWidget(title="Correlation map")
        self.corr_img = pg.ImageItem()
        self.corr_plot.addItem(self.corr_img)
        self.corr_plot.setAspectLocked(False)

        # ---- Layout ----
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self.plot, stretch=1)
        layout.addWidget(self.corr_plot, stretch=2)
        self.setCentralWidget(central)

        # ---- Worker thread ----
        self.sdr_thread = QThread()
        self.worker = SDRWorker()
        self.worker.moveToThread(self.sdr_thread)

        self.worker.end_of_run.connect(lambda: QTimer.singleShot(0, self.worker.run))
        self.worker.result_ready.connect(self.on_result)

        self.sdr_thread.started.connect(self.worker.run)
        self.sdr_thread.start()

    def on_result(self, data, corr_map, note):
        # Real-time signal
        if data.ndim > 1:
            data = data[:, 0]
        t = np.arange(len(data)) / FS - T
        self.curve.setData(t, data)

        # Correlation map: only updated when the threshold was crossed,
        # otherwise the last map stays on screen
        if corr_map is not None:
            self.corr_img.setImage(np.asarray(corr_map), autoLevels=True)
            self.corr_plot.setTitle(f"Correlation map, note: {note}")

    def closeEvent(self, event):
        self.sdr_thread.quit()
        self.sdr_thread.wait()
        super().closeEvent(event)


app = QApplication([])
window = MainWindow()
window.show()
app.exec()