import numpy as np
from MicrophoneArray import MicrophoneArray
from SoundSource import SoundSource
from parameters import constants
import math
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.io import wavfile

# Create Simulation Environment
Fs = 8000
_, audio = wavfile.read('s.wav')

audio = 0.5 * np.sum(audio, axis = 1)

# Simulate
s1 = SoundSource(np.array([[0.5, 1, 3]]))
s2 = SoundSource(np.array([[4, 1,3]]))
m = MicrophoneArray(np.array([[2,1,3]]),Fs)

signal = np.array([0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0])
s1.add_signal(signal)
s2.add_signal(signal)
sources = []
sources.append(s1)
sources.append(s2)

def free_field_propagation(sources, microphone_array):
    # segment the trajectory in a certain number of points
    # nSamples = len(source.signal)
    # nPts = source.position.shape[0]
    # nMics = microphone_array.nmic

    # fs = nSamples / nPts
    # timestamps = np.arange(nPts)

    # w_ini = np.append((timestamps*fs).astype(int), nSamples)
    # w_len = np.diff(w_ini)

    # segments = np.zeros((np.nPts, w_len.max()))
    # for n in range(nPts):
    #     segments[n, 0:w_len[n]] = source.signal[w_ini[n]:w_ini[n+1]]
    # segments = segments.astype('float32', order = 'C', copy = False)

    ## ATTEMPT 1
    # compute RIRs
    # Fractional delay length
    # fdl = constants.get("frac_delay_len")
    # fdl2 = (fdl - 1) // 2

    # dist = sources.distance(microphone_array.R)
    # time = dist / constants.get("c")

    # alpha = 1 / (4.0 * np.pi * dist)

    # N = np.ceil((1.05 * time.max()) * Fs)
    # N += fdl

    # t = np.arange(N) / float(Fs)
    # IR = np.zeros(t.shape)

    # for i in range(time.shape[0]):
    #     time_ip = int(np.round(Fs * time[i]))
    #     time_fp = (Fs * time[i]) - time_ip

    #     IR[time_ip - fdl2 : time_ip + fdl2 + 1] += alpha * fractional_delay(time_fp)

    # # plot RIR
    # plt.plot(t, IR)
    # plt.plot(t, 0.079 + np.zeros(np.shape(t)))
    # plt.show()

    rir = []
    ## ATTEMPT 2
    for m, mic in enumerate(microphone_array.R):
        rir.append([])
        for s, src in enumerate(sources):

            fdl = constants.get("frac_delay_len")
            fdl2 = fdl // 2

            dist = src.distance(mic)
            time = dist / constants.get("c")
            t_max = time.max()

            N = int(math.ceil(t_max * Fs))

            IR = np.zeros(N + fdl)
            distance_rir = np.arange(N) / Fs * constants.get("c")

            ir_loc = np.zeros_like(IR)
            alpha = 1 / (dist)
            time_adjust = time + fdl2 / Fs

            # Implementation of IS method
            ir_loc = ism_ir_builder(ir_loc, time_adjust, alpha, Fs, fdl)

            IR += ir_loc
            rir[-1].append(IR)
    
    M = len(microphone_array.R)
    S = len(sources)

    from itertools import product
    max_len_rir = np.array(
        [len(rir[i][j]) for i,j in product(range(M), range(S))]).max()
    
    L = max_len_rir
    premix_signals = np.zeros((S, M, L))

    for m in np.arange(M):
        for s in np.arange(S):
            premix_signals[s, m, 0:len(rir[m][s])] = rir[m][s]
    
    signals = np.sum(premix_signals, axis = 0)
    print("max:", rir[0][1].max())
    print("len:", len(rir[0][1]))
    plt.plot(np.arange(len(signals[0])), signals[0])
    plt.show()
    # lenIRs = 2048
    # IRs = np.zeros((nPts, nMics, lenIRs))

    # for n in range(nPts):
    #     for m in range(nMics):
    #         IRs[n, m, :] = simulatePropagation(args)
    
    # # Convolution between IR and source signal


def single_refl_propagation(sources, microphone_array):
    rir = []
    ## ATTEMPT 2
    for m, mic in enumerate(microphone_array.R):
        rir.append([])
        for s, src in enumerate(sources):

            fdl = constants.get("frac_delay_len")
            fdl2 = fdl // 2
            src.road_reflection()
            dist = np.sqrt(np.sum(np.subtract(src.images, mic)**2, axis=1))
            time = dist / constants.get("c")
            t_max = time.max()

            N = int(math.ceil(t_max * Fs))

            IR = np.zeros(N + fdl)
            distance_rir = np.arange(N) / Fs * constants.get("c")

            ir_loc = np.zeros_like(IR)
            alpha = 1 / (dist)
            time_adjust = time + fdl2 / Fs

            # Implementation of IS method
            ir_loc = ism_ir_builder(ir_loc, time_adjust, alpha, Fs, fdl)

            IR += ir_loc
            rir[-1].append(IR)
    
    M = len(microphone_array.R)
    S = len(sources)

    from itertools import product
    max_len_rir = np.array(
        [len(rir[i][j]) for i,j in product(range(M), range(S))]).max()
    
    L = max_len_rir
    premix_signals = np.zeros((S, M, L))

    for m in np.arange(M):
        for s in np.arange(S):
            premix_signals[s, m, 0:len(rir[m][s])] = rir[m][s]
    
    signals = np.sum(premix_signals, axis = 0)
    print("max:", rir[0][0].max())
    print("len:", len(rir[0][0]))
    plt.plot(np.arange(len(signals[0])), signals[0])
    plt.show()

def free_field_signal(sources, microphone_array):
    rir = []
    
    for m, mic in enumerate(microphone_array.R):
        rir.append([])
        for s, src in enumerate(sources):

            fdl = constants.get("frac_delay_len")
            fdl2 = fdl // 2

            dist = src.distance(mic)
            time = dist / constants.get("c")
            t_max = time.max()

            N = int(math.ceil(t_max * Fs))

            IR = np.zeros(N + fdl)
            distance_rir = np.arange(N) / Fs * constants.get("c")

            ir_loc = np.zeros_like(IR)
            alpha = 1 / (dist)
            time_adjust = time + fdl2 / Fs

            # Implementation of IS method
            ir_loc = ism_ir_builder(ir_loc, time_adjust, alpha, Fs, fdl)

            IR += ir_loc
            rir[-1].append(IR)
    
    M = len(microphone_array.R)
    S = len(sources)

    from itertools import product
    max_len_rir = np.array(
        [len(rir[i][j]) for i,j in product(range(M), range(S))]).max()
    
    f = lambda i: len(sources[i].signal) 

    max_sig_len = np.array([f(i) for i in range(S)]).max()
    L = int(max_len_rir) + int(max_sig_len) - 1
    if L % 2 == 1:
        L += 1

    premix_signals = np.zeros((S, M, L))

    for m in np.arange(M):
        for s in np.arange(S):
            sig = sources[s].signal
            if sig is None:
                continue
            h = rir[m][s]
            premix_signals[s, m, 0:len(sig) + len(h) - 1] += fftconvolve(h, sig)
    
    signals = np.sum(premix_signals, axis = 0)
    print("max:", rir[0][0].max())
    print("len:", len(rir[0][0]))
    plt.plot(np.arange(len(signals[0])), signals[0])
    plt.show()
    # lenIRs = 2048
    # IRs = np.zeros((nPts, nMics, lenIRs))

    # for n in range(nPts):
    #     for m in range(nMics):
    #         IRs[n, m, :] = simulatePropagation(args)
    
    # # Convolution between IR and source signal

def single_refl_signal(sources, microphone_array):
    rir = []
    
    for m, mic in enumerate(microphone_array.R):
        rir.append([])
        for s, src in enumerate(sources):

            fdl = constants.get("frac_delay_len")
            fdl2 = fdl // 2

            src.road_reflection()
            dist = np.sqrt(np.sum(np.subtract(src.images, mic)**2, axis=1))
            time = dist / constants.get("c")
            t_max = time.max()

            N = int(math.ceil(t_max * Fs))

            IR = np.zeros(N + fdl)
            distance_rir = np.arange(N) / Fs * constants.get("c")

            ir_loc = np.zeros_like(IR)
            alpha = 1 / (dist)
            time_adjust = time + fdl2 / Fs

            # Implementation of IS method
            ir_loc = ism_ir_builder(ir_loc, time_adjust, alpha, Fs, fdl)

            IR += ir_loc
            rir[-1].append(IR)
    
    M = len(microphone_array.R)
    S = len(sources)

    from itertools import product
    max_len_rir = np.array(
        [len(rir[i][j]) for i,j in product(range(M), range(S))]).max()
    
    f = lambda i: len(sources[i].signal) 

    max_sig_len = np.array([f(i) for i in range(S)]).max()
    L = int(max_len_rir) + int(max_sig_len) - 1
    if L % 2 == 1:
        L += 1

    premix_signals = np.zeros((S, M, L))

    for m in np.arange(M):
        for s in np.arange(S):
            sig = sources[s].signal
            if sig is None:
                continue
            h = rir[m][s]
            premix_signals[s, m, 0:len(sig) + len(h) - 1] += fftconvolve(h, sig)
    
    signals = np.sum(premix_signals, axis = 0)
    # print("max:", rir[0][0].max())
    # print("len:", len(rir[0][0]))
    plt.plot(np.arange(len(signals[0])), signals[0])
    plt.show()

def fractional_delay(delay):
    N = constants.get("frac_delay_len")

    return np.hanning(N) * np.sinc(np.arange(N) - (N - 1) / 2 - delay)

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


## TEST
# free_field_propagation(sources,m)
# single_refl_propagation(sources, m)
single_refl_signal(sources, m)
