from turtle import position
import numpy as np
import math
class Material:
    def __init__(
        self,
        absorption_coefficients,
        ) -> None:
        pass

class Environment:

    def __init__(
        self,
        fs = 8000,
        source = None,
        noise_sources = None,
        background_noise = None,
        mic_array = None,
        temperature = None,
        pressure = None,
        rel_humidity = None,
        road_material = None
        ) -> None:
        
        pass
    
    # Select a material from provided options in a dictionary.
    def choose_road_material():
        pass

    # Create a new road material from a set of center frequs and absorption coefficients.
    # Extrapolate to cover full frequency spectrum.
    def create_road_material(self, center_freqs, abs_coeffs):
        pass
    
    # Add sound source to envrionment in specified position P, with specified signal S and trajectory T
    # If no signal is specified, assign default one (####TBD####)
    # If no trajectory is specified, source is assumed to be static
    def add_source(self, position, signal = None, trajectory = None) -> None:
        pass
    
    # Add sound source to envrionment in specified position P, with specified signal S and trajectory T
    # If no signal is specified, assign default one (####TBD####)
    # If no trajectory is specified, source is assumed to be static
    def add_noise_source(self, position, signal = None, trajectory = None) -> None:
        pass

    # Add background noise to the simulation with a given SNR. 
    # SNR is computed w.r.t the the signal received when the source is closest to the microphone
    def add_background_noise(self, signal, SNR):
        pass

    # Add microphone array in the environment. Array contains R microphones, specified by their position in x,y,z coordinates
    def add_microphone_array(self, mic_locs: np.array) -> None:
        pass

    # plots 3D Simulation Environment, containing road surface, microphone positions and source trajectory
    def plot_environment():
        pass

    # Compute air absorption coefficients and store them in a np.array
    # Coeffs are computed based on temperature, pressure, relative humidity and are stored in dB scale
    def compute_air_absorption_coefficients():
        pass

    # Runs simulation. Returns array np.array([M,N]), where M is the number of microphones 
    # and N is the number of samples of the simulation. Array contains signals recorded by microphones
    def simulate() -> np.array:
        pass


class SoundSource:

    def __init__(
            self,
            position,
            image = None,
            signal = None,
            trajectory = None,
            fs = 8000
        ) -> None:

        self.position = position
        self.trajectory = trajectory
        self.image = image
        self.signal = signal
        self.fs = fs

    # Create trajectory given a set of N points and (N-1) velocities (assumed constant between 2 subsequent points). If 
    # a single speed value is given, speed is assumed to be constant through all trajectory
    # Returns array containing a position for each sample of the simulation.
    def create_trajectory(self, positions: np.array, speed: np.array):
        trajectory = np.array([])
        if len(speed) != (np.shape(positions)[0] - 1):
            if(len(speed) != 1):
                raise ValueError('Speed must be a float or an array with len(speed) = np.shape(positions)[0] - 1!')
            else:
                speed = speed * np.ones(np.shape(positions)[0], 1)
        for i in range(1, np.shape(positions)[0]):
            a = positions[i - 1]
            b = positions[i]
            # Direction defining straight line passing for a and b
            direction = b - a
            direction = direction / np.linalg.norm(direction)

            len_segment = np.sqrt(np.sum((a-b)**2))         # Compute length of segment A,B
            t_segment = len_segment / speed[i-1]            # Time to go from A to B (seconds)
            samples_segment = round(t_segment * self.fs)    # Number of samples to go from A to B

            segment_positions = len_segment / samples_segment * range(samples_segment)  # Positions on segment at each sample
            segment_positions = np.append(segment_positions, 5).reshape(-1,1)
            trajectory = np.tile(a,(len(segment_positions), 1)) + segment_positions * direction
            trajectory = np.append(trajectory, segment_positions)
        
        trajectory = np.append(trajectory, positions[-1,:])
        return trajectory

class SimulatorManager:

    def __init__(
        self,
        environment,
        primaryDelLine,
        secondaryDelLine,
        airAbsorptionFilters,
        asphaltReflectionfilter,

        ) -> None:
        pass
    
    # Compute initial delay and set it into the two delay lines. 
    # Compute air absorption filters and asphalt reflection filter and store the precomputed values in a table. 
    # Instantiate delay lines.
    def initialize():
        pass

    # Compute new delays given acutal positions of src and microphone.
    # Update coefficients of filters with new positions.
    # Produce output value.
    def update():
        pass
    
    # Compute air absorption FIR filter with ntaps. Depends on distance and air absorption coefficients.
    def compute_air_absorption_filter(self, abs_coeffs, distance, numtaps):
        pass

    # Compute sound attenuation depending on distance between source and receiver. 
    # Propagation is assumed to be based on spherical waves.
    def compute_sound_attenduation(self, distance):
        pass
    
    # Compute angle dependent asphalt reflection filters for a set of angles [-89,89]deg, with 
    # a resultion of 1deg, based on normal incidence asborption coefficients computed at frequencies freqs.
    # Filters are stored in a table np.array([89*2+1, ntaps]), where ntaps is the desired filter length
    def compute_asphalt_reflection(self, absorption_coeffs, freqs, ntaps):
        pass


class MicrophoneArray:
    def __init__(
            self,
            mic_positions,
        ) -> None:
        
        self.mic_positions = np.array(mic_positions)
        self.nmics = np.shape(mic_positions)[0]
        self.signals = None

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

# Compute length of segment A,B
a = np.array([0,0])
b = np.array([3,4])
direction = b - a
direction = direction / np.linalg.norm(direction)
len_segment = np.sqrt(np.sum((a-b)**2))
t_segment = len_segment / 1.3
samples_segment = round(t_segment * 11)
segment_positions = len_segment / samples_segment * range(samples_segment)
segment_positions = np.append(segment_positions, 5).reshape(-1,1)
trajectory = np.tile(a,(len(segment_positions), 1)) + segment_positions * direction
# print(segment_positions.dot(np.tile(direction, (np.shape(segment_positions)[0],1))))
# for i in range(0, len(segment_positions)):
print(trajectory)
# print(segment_positions)