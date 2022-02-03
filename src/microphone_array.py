import numpy as np

class MicrophoneArray:
    def __init__(self, R):
        self.R = np.array(R)
        self.nmics = np.shape(R)[0]
        self.signals = None
        self.center = np.mean(R, axis = 1, keepdims = True)
