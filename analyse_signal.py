import numpy as np

def analyse_signal(signal, matrice):
    corr = (matrice @ signal) / (np.linalg.norm(matrice) * np.linalg.norm(signal))
    return 


