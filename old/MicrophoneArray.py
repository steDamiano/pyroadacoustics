import numpy as np

class MicrophoneArray:
    def __init__(self, R, fs, directivity = None):
        # R contains the coordinates of the MIC array
        R = np.array(R)
        self.dim = R.shape[1]
        self.nmic = R.shape[0]

        self.R = R
        self.fs = fs
        self.directivity = directivity
        self.signals = None
        self.center = np.mean(R, axis = 1, keepdims = True)

    def record(self, signals):
        if signals.shape[0] != self.nmic:
            raise NameError("The signals array should have as many lines as there are microphones.")
        
        if signals.ndim != 2:
            raise NameError("The signals should be a 2D array.")
        
        self.signals = signals

    def to_wav(self, filename, mono = False, bitdepth = 16, norm = False):
        from scipy.io import wavfile

        if mono is True:
            signal = self.signals[self.nmic // 2]
        else:
            signal = self.signals.T

        bits = bitdepth

        signal = np.array(signal, dtype=bitdepth)

        wavfile.write(filename, self.fs, signal)
    

# R = np.array([[1, 1], [1, 1.1], [1, 1.2], [1, 1.4]])
# signals = np.array([[1,2,3,4,5],[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3]])
# fs = 8000

# mics = MicrophoneArray(R, fs)
# mics.record(signals, fs)
# print(mics.signals)