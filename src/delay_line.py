import numpy as np

class DelayLine:
        
    def __init__(self, N = 48000, write_ptr = 0, read_ptr = 0, fs = None) -> None:
        self.N = N
        if fs == None:
            print('Warning: no values detected for fs. Assigning the default fs = 8000Hz')
            fs = 8000
        self.fs = fs
        self.write_ptr = write_ptr
        self.read_ptr = read_ptr
        self.delay_line = np.zeros(N)

    def read_interpolate(self, read_ptr_integer, d, method = 'Linear'):

        # Sinc Interpolation -> Windowed Sinc function
        # h_sinc = frac_delay_sinc(a)
        
        # y2 = 0
        # for i in range(0, sinc_samples):
        #     y2 = y2 + h_sinc[i]*A[np.mod(rpi + i - math.floor(sinc_samples/2), len(A))]
        
        # # Lagrange Interpolation
        # order = 5
        # h_lagrange = frac_delay_lagrange(order, a)
        # y3 = 0

        # for i in range(0, len(h_lagrange)):
        #     y3 = y3 + h_lagrange[i] * A[np.mod(rpi + i, len(A))]

        if method == 'Sinc':
            print('Sinc Interpolation')
            pass
        if method == 'Lagrange':
            print('Lagrange Interpolation')
            pass
        if method == 'Linear':
            return d * self.delay_line[read_ptr_integer + 1] + (1 - d) * self.delay_line[read_ptr_integer]

    def update_delay_line(self, x, tau):

        # Append sample at the end of the delay line and increment write pointer
        self.delay_line[self.write_ptr] = x
        self.write_ptr += 1

        # Compute interpolated read position (fractional delay)
        rpi = np.floor(self.read_ptr)
        frac_del = self.read_ptr - rpi
        
        # Produce output with interpolated read
        y = self.read_interpolate(rpi, frac_del, 'Linear')

        # Update delay in samples and read pointer position
        M = tau * self.fs
        self.read_ptr = self.write_ptr - M

        # Check read and write pointers within delay line length
        if (self.read_ptr < 0):
            self.read_ptr += self.N
        if (self.write_ptr >= self.N - 1):
            self.write_ptr -= self.N
        if (self.read_ptr >= self.N - 1):
            self.read_ptr -= self.N
        
        return y

# Update delay
# mic.dproj_refl -= src.v / self.fs

# d2 = dline2 + dproj_refl ** 2
# # d = np.sqrt(d2)
# tau = drefl / c
# M = tau * fs
# rptrB = wptrB - M
# delays_sec_refl2.append(tau)
# if (rptrB < 0):
#     rptrB += N
# if (wptrB >= N - 1):
#     wptrB -= (N)
# if (rptrB >= N - 1):
#     rptrB -= (N)
# return y1, d`


## TEST CODE
test = DelayLine()
test.update_delay_line(1, 0)