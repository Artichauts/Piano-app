# Vous devez installer la librairie sounddevice
# PIP :  pip install sounddevice
# Anaconda : conda install -c conda-forge python-sounddevice

import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np

def record_audio(seconds, fs):
    """Enregistre l'audio pendant un certain nombre de secondes.
    seconds : période d'enregistement 
    fs : fréquence d'acquisition
    """ 
    
    default = True #Si cette option est utilisée, le micro/speaker par défaut est utilisé
    devices = sd.query_devices()

    if not default:
        InputStr = "Choisir le # correspondant au micro parmis la liste: \n"
        OutputStr = "Choisir le # correspondant au speaker parmis la liste: \n"
        for i in range(len(devices)):
            if devices[i]['max_input_channels']:
                InputStr += ('%d : %s \n' % (i, ''.join(devices[i]['name'])))
            if devices[i]['max_output_channels']:
                OutputStr += ('%d : %s \n' % (i, ''.join(devices[i]['name'])))
        DeviceIn = input(InputStr)
        DeviceOut = input(OutputStr)

        sd.default.device = [int(DeviceIn), int(DeviceOut)]

    print("Recording with : {} \n".format(devices[sd.default.device[0]]['name']))
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels= 1)
    sd.wait()
    print("Recording finished.")

    return myrecording 







def plot_fft(signal, fs, fmax=None, db=False, window=True):
    """Trace le spectre d'amplitude (FFT) d'un signal.
    signal : tableau 1D ou (N, 1) de l'enregistrement
    fs     : fréquence d'acquisition [Hz]
    fmax   : fréquence max à afficher [Hz] (None = Nyquist)
    db     : True pour afficher en dB
    window : applique une fenêtre de Hann pour réduire la fuite spectrale
    """
    x = np.asarray(signal).flatten()      # (N, 1) -> (N,)
    x = x - np.mean(x)                    # retire la composante DC
    N = len(x)

    if window:
        w = np.hanning(N)
        x = x * w
        norm = np.sum(w) / 2              # correction d'amplitude de la fenêtre
    else:
        norm = N / 2

    X = np.fft.rfft(x)                    # spectre unilatéral (signal réel)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    mag = np.abs(X) / norm                # amplitude du sinus équivalent

    if db:
        mag = 20 * np.log10(mag + 1e-12)

    plt.figure()
    plt.plot(freqs, mag)
    plt.xlabel('Fréquence [Hz]')
    plt.ylabel('Amplitude [dB]' if db else 'Amplitude')
    plt.title('FFT du signal')
    plt.xlim(0, fmax if fmax else fs / 2)
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    
    seconds = 5 
    fs = 150000      # Sampling rate    

    myrecording = record_audio(seconds, fs)

    print("Shape of the recording: ", myrecording.shape)
    print("Data type of the recording: ", myrecording.dtype)
    print("myrecording: ", myrecording)

    t = np.arange(0,5,1/fs)

    plt.plot(t, myrecording)
    plt.xlabel('Temps [s]')
    plt.ylabel('Amplitude')
    plt.show()

    np.savetxt("myrecording.csv", myrecording, delimiter=",")
    #plot_fft(myrecording, fs)
