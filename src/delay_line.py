import numpy as np
import math

### OLD CODE
# class DelayLine:
        
#     def __init__(self, N = 48000, write_ptr = 0, read_ptr = 0, fs = None) -> None:
#         self.N = N
#         if fs == None:
#             print('Warning: no values detected for fs. Assigning the default fs = 8000Hz')
#             fs = 8000
#         self.fs = fs
#         self.write_ptr = write_ptr
#         self.read_ptr = read_ptr
#         self.delay_line = np.zeros(N)
    
#     # Set Delays for Multiple Read Pointers
#     def set_delays(self, delay):
#         for i in range(len(self.read_ptr)):
#             self.read_ptr[i] = self.write_ptr - delay[i]
#             if (self.read_ptr[i] < 0):
#                 self.read_ptr[i] += self.N

#     # Set Delay for Single Read Pointer
#     def set_delay_single(self, delay):
#         self.read_ptr = self.write_ptr - delay
#         if (self.read_ptr < 0):
#             self.read_ptr += self.N
    
#     def read_interpolate(self, read_ptr_integer, d, method = 'Linear'):

#         # Sinc Interpolation -> Windowed Sinc function
#         # h_sinc = frac_delay_sinc(a)
        
#         # y2 = 0
#         # for i in range(0, sinc_samples):
#         #     y2 = y2 + h_sinc[i]*A[np.mod(rpi + i - math.floor(sinc_samples/2), len(A))]
        
#         # # Lagrange Interpolation
#         # order = 5
#         # h_lagrange = frac_delay_lagrange(order, a)
#         # y3 = 0

#         # for i in range(0, len(h_lagrange)):
#         #     y3 = y3 + h_lagrange[i] * A[np.mod(rpi + i, len(A))]

#         if method == 'Sinc':
#             print('Sinc Interpolation')
#             pass
#         if method == 'Lagrange':
#             print('Lagrange Interpolation')
#             pass
#         if method == 'Linear':
#             return d * self.delay_line[read_ptr_integer + 1] + (1 - d) * self.delay_line[read_ptr_integer]

#     def update_delay_line(self, x, tau):
#         y = np.zeros_like(self.read_ptr)
#         # Append sample at the end of the delay line and increment write pointer
#         self.delay_line[self.write_ptr] = x
#         self.write_ptr += 1
#         for i in range(len(self.read_ptr)):
#             # Compute interpolated read position (fractional delay)
#             rpi = int(np.floor(self.read_ptr[i]))
#             frac_del = self.read_ptr[i] - rpi
        
#             # Produce output with interpolated read
#             y[i] = self.read_interpolate(rpi, frac_del, 'Linear')

#             # Update delay in samples and read pointer position
#             M = tau[i] * self.fs
#             self.read_ptr[i] = self.write_ptr - M

#             # Check read and write pointers within delay line length
#             while (self.read_ptr[i] < 0):
#                 self.read_ptr[i] += self.N
#             while (self.write_ptr >= self.N - 1):
#                 self.write_ptr -= self.N
#             while (self.read_ptr[i] >= self.N - 1):
#                 self.read_ptr[i] -= self.N
        
#         return y

# # Update delay
# # mic.dproj_refl -= src.v / self.fs

# # d2 = dline2 + dproj_refl ** 2
# # # d = np.sqrt(d2)
# # tau = drefl / c
# # M = tau * fs
# # rptrB = wptrB - M
# # delays_sec_refl2.append(tau)
# # if (rptrB < 0):
# #     rptrB += N
# # if (wptrB >= N - 1):
# #     wptrB -= (N)
# # if (rptrB >= N - 1):
# #     rptrB -= (N)
# # return y1, d`


# ## TEST CODE
# # test = DelayLine()
# # print(test.read_ptr)
# # print(test.write_ptr)
# # test.set_delay_single(10)
# # print("Delay has been set")
# # print(test.read_ptr)
# # print(test.write_ptr)



class DelayLine:
    def __init__(
            self, 
            N = 48000, 
            write_ptr = 0, 
            read_ptr = 0, 
            fs = None,
            interpolation = 'Linear',
        ):

        self.N = N
        if fs == None:
            print('Warning: no values detected for fs. Assigning the default fs = 8000Hz')
            fs = 8000
        self.fs = fs
        self.write_ptr = write_ptr
        self.read_ptr = read_ptr
        self.delay_line = np.zeros(N)
        self.interpolation = interpolation

    # Set initial delays -> position of readpointers w.r.t. write pointer position
    def set_delays(self, delay):
        for i in range(len(self.read_ptr)):
            self.read_ptr[i] = self.write_ptr - delay[i]
            if (self.read_ptr[i] < 0):
                self.read_ptr[i] += self.N
  
    # Advance position of write pointer (1 sample increase) and write x on delay line
    # Compute position of read pointers given delays tau (N delays, one per each write pointer)
    # Produce output by computing interpolated read
    def update_delay_line(self, x, tau):
        y = np.zeros_like(self.read_ptr)
        # Append sample at the end of the delay line and increment write pointer
        self.delay_line[self.write_ptr] = x
        self.write_ptr += 1
        for i in range(len(self.read_ptr)):
            # Compute interpolated read position (fractional delay)
            rpi = int(np.floor(self.read_ptr[i]))
            frac_del = self.read_ptr[i] - rpi
        
            # Produce output with interpolated read
            y[i] = self._interpolated_read(rpi, frac_del, self.interpolation)

            # Update delay in samples and read pointer position
            M = tau[i] * self.fs
            self.read_ptr[i] = self.write_ptr - M

            # Check read and write pointers within delay line length
            while (self.read_ptr[i] < 0):
                self.read_ptr[i] += self.N
            while (self.write_ptr >= self.N - 1):
                self.write_ptr -= self.N
            while (self.read_ptr[i] >= self.N - 1):
                self.read_ptr[i] -= self.N
        
        return y

    # Produce output with interpolated delay line read
    # Parameters: integer and fractional part of delay, interpolation method
    def _interpolated_read(self, read_ptr_integer, d, method = 'Linear'):
        if method == 'Lagrange':
            order = 5
            h_lagrange = self._frac_delay_lagrange(order, d)
            print('Lagrange Interpolation, order %d' %order)
            
            out = 0
            for i in range(0, len(h_lagrange)):
                out = out + h_lagrange[i] * self.delay_line[np.mod(read_ptr_integer + i, self.N)]

        elif method == 'Linear':
            return d * self.delay_line[read_ptr_integer + 1] + (1 - d) * self.delay_line[read_ptr_integer]
        else:
            sinc_samples = 81
            sinc_window = np.hanning(sinc_samples)
            print('Sinc Interpolation, samples: %d' %sinc_samples)
            # Sinc Interpolation -> Windowed Sinc function
            h_sinc = self._frac_delay_sinc(sinc_samples, sinc_window, d)
                
            out = 0
            for i in range(0, sinc_samples):
                out = out + h_sinc[i]*self.delay_line[np.mod(read_ptr_integer + i - math.floor(sinc_samples/2), self.N)]
            
    
    # Compute fractional delay filter using Lagrange polynomial interpolation method
    def _frac_delay_lagrange(self, order, delay):
        n = np.arange(0, order + 1)
        h = np.ones(order + 1)

        for k in range(0, order + 1):
            # Find index n != k
            index = []
            for j in range(0, order + 1):
                if j != k:
                    index.append(j)
            
            h[index] = h[index] * (delay - k) / (n[index] - k)
        return h
    
    # Windowed Sinc Function
    def _frac_delay_sinc(self, sinc_samples, sinc_window, delay):
        return sinc_window * np.sinc(np.arange(0,sinc_samples) - (sinc_samples - 1) / 2 - delay)



test2 = DelayLine(N = 48000, write_ptr=0, read_ptr=np.array([0,0]), fs = 8000)
print(test2.read_ptr)
test2.set_delays(np.array([10,40]))
print(test2.read_ptr)
test2.update_delay_line(10, np.array([0.01,0.004]))
print(test2.read_ptr)