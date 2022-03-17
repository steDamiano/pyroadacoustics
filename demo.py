import pyroadacoustics as pyroad
import numpy as np
import scipy.signal

from scipy.io import wavfile
samplerate, data = wavfile.read('siren.wav')
data = scipy.signal.resample(data, int(len(data) * 8000 / samplerate))

data = np.reshape(data[:,0], (-1, 1))
data = data / max(data)
data = np.reshape(data, -1)
print(np.shape(data))


samplerate, noise = wavfile.read('ambient.wav')
noise = scipy.signal.resample(noise, int(len(noise) * 8000 / samplerate))

noise = np.reshape(noise[44100:120000,0], (-1, 1))
noise = noise / max(noise)
noise = np.reshape(noise, -1)
noise = noise[16000:]


fs = 8000
t = np.arange(0,5,1/fs)
f = 2000
env = pyroad.Environment(fs = fs)
# src_signal = np.sin(2 * np.pi * f * t)
src_signal = data
# src_signal = np.random.randn(len(t))
env.add_source(np.array([3,5,1]), trajectory_points=np.array([[3,40,1], [3,-40,1]]), source_velocity=np.array([5]), signal=src_signal)
env.add_microphone_array(np.array([[0,0,1]]))
# env.set_background_noise(noise, SNR=0)
env.set_simulation_params("Allpass", True, True)
import time
start_time = time.time()
signal = env.simulate()
print(time.time() - start_time)

# Run simulation and plot signals received at microphone array
import matplotlib.pyplot as plt
# plt.plot(np.arange(len(signal[0]))/fs, signal[0])
# plt.title('Received Signal')
# plt.xlabel('Time [s]')

import scipy.signal
from scipy.io.wavfile import write

plt.figure()
ff, tt, Sxx = scipy.signal.spectrogram(signal[0], fs = fs)
plt.pcolormesh(tt, ff, Sxx, shading='auto', vmax = 0.000002)
plt.title('Spectrogram')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.show()

write('output.wav', fs, signal[0])


# Nfft = 2048
# f_axis = np.linspace(0, fs / 2, int(Nfft / 2 + 1))
# fft_signal = np.fft.fft(signal[0], Nfft)
# plt.figure()
# plt.plot(f_axis, abs(fft_signal[:int(Nfft/2 + 1)]))
