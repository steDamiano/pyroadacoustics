import numpy as np

def frac_delay_sinc(delay, num_samples, window = None):
    if (window is None):
        window = np.hanning(num_samples)
    if (len(window) != num_samples):
        raise ValueError('Window Length must be equal to Number of Samples')
    
    return window * np.sinc(np.arange(0,num_samples) - (num_samples - 1) / 2 - delay)

def frac_delay_lagrange(N, delay):
    n = np.arange(0,N+1)
    h = np.ones(N+1)

    for k in range(0,N+1):
        index = []
        for j in range(0,N+1):
            if j != k:
                index.append(j)
        
        h[index] = h[index] * (delay - k) / (n[index] - k)

    return h