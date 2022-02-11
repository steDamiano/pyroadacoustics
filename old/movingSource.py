import numpy as np
from MicrophoneArray import MicrophoneArray
from SoundSource import SoundSource
from parameters import constants
from scipy.signal import fftconvolve
from scipy.io import wavfile
import matplotlib.pyplot as plt
import math
import time as ttime

Fs = 8000
c = constants.get("c")

nPts = 128
pos_src = np.tile(np.array([0.0, 3.0, 1]), (nPts, 1))
pos_src[:,0] = np.linspace(0.0, 12, nPts)

m = MicrophoneArray(np.array([[5,4,1]]),Fs)

## SINGLE SOURCE
s1 = SoundSource(np.array([0.0, 3.0, 1.0]))
# signal = np.array([0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0])

# Define sinusoid
f = 200
time = np.arange(0, 6, 1/Fs)
signal = np.sin(2 * np.pi * f * time)
# signal = np.random.randn(len(time))
check, audio = wavfile.read("s.wav")
audio = 0.5 * np.sum(audio, axis = 1)
aug_audio = np.concatenate((audio, np.zeros(48000 - len(audio))))
s1.add_signal(aug_audio)

def compute_trajectory(src, mic_array, positions, lenRIR = 256):
    # Compute RIRs between sources and receivers at every position
    nPts = len(positions)
    nMics = mic_array.nmic
    sigLen = len(src.signal)

    print("nPts: ", nPts)
    print("nMics: ", nMics)
    print("Length Signal: ", sigLen)

    RIRs = np.zeros((nPts, nMics, lenRIR))
    print(np.shape(RIRs))

    # Compute RIRs
    RIRs = compute_rir(positions, mic_array, lenRIR)
    
    
    # Segment the signal and convolve with the RIRs
    fPts = sigLen / nPts
    timestamps = np.arange(nPts)

    w_ini = np.append((timestamps * fPts).astype(int), sigLen)
    w_len = np.diff(w_ini)

    # OLA procedure
    segments = np.zeros((nPts, 2 * w_len.max()))
    hop_size = w_len.max()
    flag = False
    n = 0
    win = np.hanning(2 * w_len.max())
    filtered_signal = np.zeros((sigLen+lenRIR+81-1, nMics))

    for n in range(nPts-1):
        segments[n, 0:2 * w_len[n]] = win * src.signal[w_ini[n]:w_ini[n+2]]
        segments = segments.astype('float32', order='C', copy=False)
        convolution = fftconvolve(segments[n], RIRs[0][n])
        filtered_signal[w_ini[n] : w_ini[n+2]+lenRIR+81-1, 0] += convolution



    # for n in range(nPts):
    #     segments[n, 0:w_len[n]] = src.signal[w_ini[n]:w_ini[n+1]]
    #     # segments[n, 0:w_len[n]] = np.hanning(w_len[n]) * segments[n, 0:w_len[n]]

    # segments = segments.astype('float32', order='C', copy=False)

    # # Compute final microphone signal using OLA procedure
    # convolution = np.zeros((nPts, w_len.max() + lenRIR + 81 - 1))

    # for n in range(nPts):
    #     convolution[n] = fftconvolve(segments[n], RIRs[0][n])

    # filtered_signal = np.zeros((sigLen+lenRIR+81-1, nMics))

    # for m in range(nMics):
    #     for n in range(nPts):
    #         filtered_signal[w_ini[n] : w_ini[n+1]+lenRIR+81-1, m] += convolution[n, 0:w_len[n]+lenRIR+81-1]
    
    plt.plot(range(0,len(filtered_signal)),filtered_signal)

    plt.show()
    wavfile.write('prova.wav', Fs, 0.0002*filtered_signal)
    
def compute_rir(sources, microphone_array, lenRIR = None):
    rir = []
    for m, mic in enumerate(microphone_array.R):
        rir.append([])
        for s, srcpos in enumerate(sources):
            src = SoundSource(np.array([srcpos]))
            fdl = constants.get("frac_delay_len")
            fdl2 = fdl // 2
            # src.road_reflection()
            dist = np.sqrt(np.sum(np.subtract(src.images, mic)**2, axis=1))
            time = dist / constants.get("c")
            t_max = time.max()
            if lenRIR == None:
                N = int(math.ceil(t_max * Fs))
            else:
                N = lenRIR

            IR = np.zeros(N + fdl)
            # distance_rir = np.arange(N) / Fs * constants.get("c")

            ir_loc = np.zeros_like(IR)
            alpha = 1 / (dist)
            time_adjust = time + fdl2 / Fs

            # Implementation of IS method
            ir_loc = ism_ir_builder(ir_loc, time_adjust, alpha, Fs, fdl)
            samp_diff = N + fdl - len(ir_loc)
            if samp_diff > 0:
                ir_loc.append(np.zeros(samp_diff))
            else:
                ir_loc = ir_loc[0:N + fdl]
            
            IR += ir_loc
            rir[-1].append(IR)
    return rir

def ism_ir_builder(rir, time, alpha, Fs, fdl):
    fdl2 = (fdl - 1) // 2
    n_times = time.shape[0]

    delta = 1.0 / 20
    lut_size = (fdl + 1) * 20 + 1
    n = np.linspace(-fdl2 - 1, fdl2 + 1, lut_size)

    sinc_lut = np.sinc(n)
    hann = np.hanning(fdl)

    for i in range(n_times):
        sample_frac = Fs * time[i]
        time_ip = int(math.floor(sample_frac))
        time_fp = sample_frac - time_ip

        x_off_frac = (1. - time_fp) * 20
        lut_gran_off = int(math.floor(x_off_frac))
        x_off = (x_off_frac - lut_gran_off)
        lut_pos = lut_gran_off

        k = 0
        for f in range(-fdl2, fdl2 + 1):
            rir[time_ip + f] += alpha[i] * hann[k] * (sinc_lut[lut_pos] + x_off * (sinc_lut[lut_pos + 1] - sinc_lut[lut_pos]))
            lut_pos += 20
            k += 1
    
    return rir # plt.plot(np.arange(len(rir)), rir)
start_time = ttime.time()
compute_trajectory(s1, m, pos_src)
print("--- %s seconds ---" % (ttime.time() - start_time))