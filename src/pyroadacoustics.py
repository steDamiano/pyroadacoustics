import numpy as np
import math
import scipy.signal
import json
import matplotlib.pyplot as plt

# Load materials as dictionary
with open('src/materials.json') as database_materials:
    materials_dict = json.load(database_materials)
# print('Type: ', type(materials_dict))
# print('\nMaterial1: ', materials_dict['m1_asphalt'])

class Material:
    """
    Parameters
    ----------
    absorption_coefficients: str or dict
        * str: The absorption values will be obtained from the database.
        * dict: A dictionary containing keys ``description``, ``coeffs``, and
            ``center_freqs``.
    """
    def __init__(
            self,
            absorption,
        ) -> None:
        
        if isinstance(absorption, str):
            absorption = dict(materials_dict[absorption])
        
        elif isinstance(absorption, dict):
            if "coeffs" in absorption and "center_freqs" not in absorption:
                raise KeyError("'absorption' must be a dictionary with keys 'coeffs' and 'center_freqs'")

            if len(absorption["coeffs"]) != len(absorption["center_freqs"]):
                raise KeyError("Length of 'absorption['coeffs']' and "
                    "'absorption['center_freqs']' must match.")
        else:
            raise TypeError('Wrong Material Assignment')
        
        self.absorption = absorption
        self.reflection_coeffs = np.sqrt(1 - self.absorption["coeffs"])
    
    def extrapolate_coeffs_to_spectrum(self, interp_degree = 2, fs = 8000, n_bands = 8):
        coeffs = self.absorption["coeffs"]
        freqs = self.absorption["center_freqs"]

        # Comment to remove dummy point at 8kHz
        freqs = np.append(freqs, 8000)
        coeffs = np.append(self.absorption["coeffs"], 0.3)
        
        model_abs = np.polyfit(freqs, coeffs, interp_degree)
        full_spectrum = np.linspace(0, 1, n_bands) * fs / 2
        estimated_abs_coeffs = np.polyval(model_abs, full_spectrum)
        
        # Clip between 0 and 1
        estimated_abs_coeffs[estimated_abs_coeffs < 0.0] = 0.0
        estimated_abs_coeffs[estimated_abs_coeffs > 1.0] = 1.0

        self.absorption = {
            "description": "Interpolated Material",
            "coeffs": estimated_abs_coeffs,
            "center_freqs": full_spectrum
        }
        
        # Compute reflection coefficients from absorption
        self.reflection_coeffs = np.sqrt(1 - estimated_abs_coeffs)

        return estimated_abs_coeffs

    def plot_absorption_coeffs(self):
        plt.figure()
        plt.plot(self.absorption["center_freqs"], self.absorption["coeffs"])
        plt.title('Absorption Coefficients Road Surface')
        plt.ylabel(r'$\alpha')
        plt.xlabel('f [Hz]')
        plt.show()
        

class Environment:

    def __init__(
            self,
            fs = 8000,
            c = 343,
            source = None,
            noise_sources = None,
            background_noise = None,
            mic_array = None,
            temperature = None,
            pressure = None,
            rel_humidity = None,
            road_material = None
        ) -> None:
        
        self.fs = fs
        self.c = c
        self.source = source
        self.noise_sources = noise_sources
        self.background_noise = background_noise
        self.mic_array = mic_array
        self.temperature = temperature
        self.pressure = pressure
        self.rel_humidity = rel_humidity
        self.road_material = road_material
        self.air_absorption_coefficients = self._compute_air_absorption_coefficients()
    
    # Select a material from provided options in a dictionary, given input string absorption.
    # If material does not exist, create new material from input dictionary of absorption coeffs and
    # center frequencies absorption = {"coeffs": [], "center_freqs" = []}
    def get_road_material(self, absorption):
        material = Material(absorption)
        
        # If frequencies do not cover the whole spectrum, perform interpolation
        if material.absorption["center_freqs"][-1] != self.fs / 2:
            material.extrapolate_coeffs_to_spectrum(fs = self.fs)
        
        self.road_material = material
        return material
    
    # Add sound source to envrionment in specified position P, with specified signal S and trajectory T
    # If no signal is specified, assign default one (####TBD####)
    # If no trajectory is specified, source is assumed to be static
    def add_source(self, position, signal = None, trajectory = None):
        if trajectory == None:
            # Define duration of the simulation
            simulation_duration = 5
            trajectory = np.tile(position, (simulation_duration * self.fs))
        if signal == None:
            # Define a default signal --> sinusoidal siren
            simulation_duration = len(trajectory) / self.fs
            
            ## Define sinusoid with frequency modulation
            f = 800
            f_lfo = 0.3

            t = np.arange(0, simulation_duration, 1/self.fs)
            signal = np.zeros_like(t)
            for i in range(len(t)):
                signal[i] = 0.2 * np.sin(2 * np.pi * f * t[i] + 600 * np.sin(2 * np.pi * f_lfo * t[i]))

        
        source = SoundSource(position, position - np.array([0,0, 2 * position[2]]), signal, trajectory)
        self.source = source
    
    # Add noise source to envrionment in specified position P
    # If no signal is specified, assign default one (####TBD####)
    # Noise source is assumed to be static
    def add_noise_source(self, position, signal = None) -> None:
        raise NotImplementedError('Noise Sources are not supported yet')

    # Add background noise to the simulation with a given SNR. 
    # SNR is computed w.r.t the the signal received when the source is closest to the microphone
    def add_background_noise(self, signal, SNR):        
        if self.source == None:
            raise RuntimeError("To add a background noise you need to first insert a sound source")
        
        if signal == None:
            # Define default signal --> white noise
            if self.source.trajectory == None:
                simulation_duration = 5
                self.source.trajectory = np.tile(self.source.position, (simulation_duration * self.fs, 1))
            else:
                simulation_duration = len(self.source.trajectory) / self.fs
            
            t = np.arange(0, simulation_duration, 1 / self.fs)
            signal = np.random.randn(t)

        # Find closest position between source and first microphone (reference)
        d_min = float('inf')

        for i in range(len(self.source.trajectory)):
            d_temp = np.sqrt(np.sum((self.mic_array.mic_positions[0] - self.source.trajectory[i]) ** 2))
            if  d_temp < d_min:
                d_min = d_temp
        
        # Compute SNR
        noise_attenuation = np.sqrt(np.sum((self.source.signal ** 2) / (4 * np.pi * d_min)) / (10 ** (SNR/10) * np.sum((signal ** 2) / (4 * np.pi * d_min))))
        signal = signal * noise_attenuation


    # Add microphone array in the environment. Array contains R microphones, specified by their position in x,y,z coordinates
    def add_microphone_array(self, mic_locs: np.array) -> None:
        mic_array = MicrophoneArray(mic_locs)
        self.mic_array = mic_array

    # plots 3D Simulation Environment, containing road surface, microphone positions and source trajectory
    def plot_environment(self):
        plt.figure()
        plt.plot(self.source.trajectory[:,0], self.source.trajectory[:,1])
        plt.scatter(self.mic_array.mic_positions[:,0], self.mic_array.mic_positions[:,1], color = 'green', marker='x')
        plt.legend(['Source Trajectory', 'Microphones'])
        plt.xlabel('[m]')
        plt.ylabel('[m]')
        plt.suptitle('Simulation Scenario')
        plt.title('Source and Microphones have height: z = %.2f' %self.mic_array.mic_positions[0,2])
        plt.show()

    # Compute air absorption coefficients and store them in a np.array
    # Coeffs are computed based on temperature, pressure, relative humidity and are stored in dB scale
    def _compute_air_absorption_coefficients(self):
        T0 = 293.15
        T01 = 273.16
        ps0 = 1
        f = np.linspace(0, 11000, num=50)
        Csat = -6.8346 * math.pow(T01 / self.temperature, 1.261) + 4.6151
        rhosat = math.pow(10, Csat)
        H = rhosat * self.rel_humidity * ps0 / self.pressure

        frn = (self.pressure / ps0) * math.pow(T0 / self.temperature, 0.5) * (
                9 + 280 * H * math.exp(-4.17 * (math.pow(T0 / self.temperature, 1/3.) - 1)))

        fro = (self.pressure / ps0) * (24.0 + 4.04e4 * H * (0.02 + H) / (0.391 + H))

        alpha = f * f * (
            1.84e-11 / ( math.pow(T0 / self.temperature, 0.5) * self.pressure / ps0 )
            + math.pow(self.temperature / T0, -2.5)
            * (
                0.10680 * math.exp(-3352 / self.temperature) * frn / (f * f + frn * frn)
                + 0.01278 * math.exp(-2239.1 / self.temperature) * fro / (f * f + fro * fro)
                )
            )
        
        return alpha

    # Runs simulation. Returns array np.array([M,N]), where M is the number of microphones 
    # and N is the number of samples of the simulation. Array contains signals recorded by microphones
    def simulate(self) -> np.array:
        # Duration of the simulation
        N = len(self.source.trajectory)
        M = self.mic_array.nmics

        signals = np.zeros((M,N))

        for m in range(M):
            # Select Active Microphone
            active_mic = self.mic_array.mic_positions[m]
        
            # Instantiate Simulator Manager or call instance
            manager = SimulatorManager(environment = self, active_microphone = active_mic, source = self.source, 
            airAbsorptionFilters = self.air_absorption_coefficients)

            # Define simulation loop
            _temp = manager.initialize(self.source.position, active_mic)

            # Compute output samples
            for n in range(N):
                signals[m,n] = manager.update(self.source.trajectory[n], active_mic, self.source.signal[n])


class SoundSource:

    def __init__(
            self,
            position,
            image_pos = None,
            signal = None,
            trajectory = None,
            fs = 8000
        ) -> None:

        self.position = position
        self.trajectory = trajectory
        self.image_pos = image_pos
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
    __instance = None
    _read1Buf = np.zeros(20)
    _read2Buf = np.zeros(20)
    _read3Buf = np.zeros(20)
    _read4Buf = np.zeros(20)
    _readBufPtr = 0

    @staticmethod
    def getInstance():
        if SimulatorManager.__instance == None:
            SimulatorManager()
        return SimulatorManager.__instance

    def __init__(
            self,
            environment: Environment,
            active_microphone,
            source,
            primaryDelLine,
            secondaryDelLine,
            airAbsorptionFilters,
        ) -> None:
        if SimulatorManager.__instance != None:
            raise Exception("SimulatorManager already instantiated")
        else:
            SimulatorManager.__instance = self
        self.environment = environment
        self.active_microphone = active_microphone
        self.source = source
        self.primaryDelLine = primaryDelLine
        self.secondaryDelLine = secondaryDelLine
        self.airAbsorptionFilters = airAbsorptionFilters
        
        self.asphaltReflectionFilterTable = self.compute_angle_reflection_table(
            self.environment.road_material.abs_coeffs, 
            self.environment.road_material.freqs, 
            ntaps = 10)

    # Compute initial delay and set it into the two delay lines. 
    # Compute air absorption filters and asphalt reflection filter and store the precomputed values in a table. 
    # Instantiate delay lines.
    def initialize(self, src_pos, mic_pos):
        self.primaryDelLine = DelayLine(N = 48000, write_ptr = 0, read_ptr = np.array([0,0]), fs = self.fs)
        self.secondaryDelLine = DelayLine(N = 48000, write_ptr = 0, read_ptr = np.array([0]), fs = self.fs)

        # Compute direct distance and delay
        d, tau = self._compute_delay(src_pos, mic_pos, self.environment.c)
        
        # Compute incidence angle
        theta = self._compute_angle(src_pos, mic_pos)

        # Compute distance between src and reflection point
        a = src_pos[2] / np.sin(theta)
        tau_1 = a / self.environment.c
        
        # Compute distance between reflection point and microphone
        b = src_pos[2] / np.sin(theta)
        tau_2 = b / self.environment.c

        # Set initial delays
        self.primaryDelLine.set_delays(np.array([tau, tau_1]))
        self.secondaryDelLine.set_delays(tau_2)
        

    # Compute new delays given acutal positions of src and microphone.
    # Update coefficients of filters with new positions.
    # Produce output value.
    def update(self, src_pos, mic_pos, signal_sample):
        # Compute direct distance and delay
        d, tau = self._compute_delay(src_pos, mic_pos, self.environment.c)

        # Compute incidence angle
        theta = self._compute_angle(src_pos, mic_pos)

        # Compute distance between src and reflection point
        a = src_pos[2] / np.sin(theta)
        tau_1 = a / self.environment.c
        
        # Compute distance between reflection point and microphone
        b = mic_pos[2] / np.sin(theta)
        tau_2 = b / self.environment.c

        # Update delays and get new sample reads
        y_primary = self.primaryDelLine.update_delay_line(signal_sample, np.array([tau, tau_1]))

        # Store the read samples in a circular array to be used for filtering with air abs and asphalt refl
        self._read1Buf = y_primary[0]
        self._read2Buf = y_primary[1]
        # self._read3Buf = y_secondary

        self._readBufPtr +=1
        if self._readBufPtr >= 20:
            self._readBufPtr -= 20
        
        ### DIRECT PATH ###

        # Attenuation due to distance
        att = self.compute_sound_attenduation(d)

        # Attenuation due to air absorption
        filt_coeffs = self.compute_air_absorption_filter(self.airAbsorptionFilters, d, numtaps = 10)
        sample_eval = 0
        for ii in range(10):
            sample_eval = sample_eval + self._read1Buf[self._readBufPtr - ii] * filt_coeffs[ii]
        
        # Direct Path Output Sample
        y_dir =  att * sample_eval

        ### REFLECTED PATH ###

        # 1. From Source to Road Surface

        # Attenuation due to distance
        att = self.compute_sound_attenduation(a)
        
        # Attenuation due to air absorption
        filt_coeffs = self.compute_air_absorption_filter(self.airAbsorptionFilters, a, numtaps = 10)

        sample_eval = 0
        for ii in range(10):
            sample_eval = sample_eval + self._read2Buf[self._readBufPtr - ii] * filt_coeffs[ii]
        sample_eval = att * sample_eval
        self._read3Buf[self._readBufPtr] = sample_eval

        # 2. Asphalt Absorption
        asphalt_filter_coeffs = self.compute_asphalt_reflection_filter(90 - theta, 
            self.asphaltReflectionFilterTable, np.array([-89, 89]))
        
        sample_eval = 0
        for ii in range(10):
            sample_eval = sample_eval + self._read3Buf[self._readBufPtr - ii] * asphalt_filter_coeffs[ii]
        
        
        # 3. Second path in air --> Secondary Delay Line
        y_secondary = self.primaryDelLine.update_delay_line(sample_eval, np.array([tau_2]))
        self._read4Buf[self._readBufPtr] = y_secondary
        # 4. From Road Surface to Receiver

        # Attenuation due to distance
        att = self.compute_sound_attenduation(b)

        # Attenuation due to air absorption
        # alpha = compute_air_absorption_coeffs(T, p_s, hrar, F)
        # alpha = 10 ** (-alpha * drefl / 20)     # Convert coeffs in dB to linear scale
        filt_coeffs = self.compute_air_absorption_filter(self.airAbsorptionFilters, b, numtaps = 10)

        sample_eval = 0
        for ii in range(10):
            sample_eval = sample_eval + self._read4Buf[self._readBufPtr - ii] * filt_coeffs[ii]
        y_refl = att * sample_eval

        y_received = y_dir + y_refl

        return y_received

    
    # Compute air absorption FIR filter with ntaps. Depends on distance and air absorption coefficients.
    def compute_air_absorption_filter(self, abs_coeffs, distance, numtaps):
        f = np.linspace(0, 11000, num=50)
        norm_freqs = f / max(f)
        alpha = 10 ** (-abs_coeffs * distance / 20)     # Convert coeffs in dB to linear scale
        filt_coeffs = scipy.signal.firwin2(numtaps, norm_freqs, alpha)
        
        return filt_coeffs

    # Compute sound attenuation depending on distance between source and receiver. 
    # Propagation is assumed to be based on spherical waves.
    def compute_sound_attenduation(self, distance):
        return 1 / (4 * np.pi * distance)
    
    # Compute angle dependent asphalt reflection filters for a set of angles [-89,89]deg, with 
    # a resultion of 1deg, based on normal incidence asborption coefficients computed at frequencies freqs.
    # Filters are stored in a table np.array([89*2+1, ntaps]), where ntaps is the desired filter length
    def compute_angle_reflection_table(self, absorption_coeffs, freqs, ntaps):
        theta_vector = np.arange(-89,89,1)
        Z0 = self._compute_air_impedance()
        # c = self.environment._compute_speed_sound()
        refl = np.sqrt(1 - absorption_coeffs)
        Z = - Z0 * (refl + 1) / (refl-1)
        b_fir = np.zeros((len(theta_vector), ntaps))
        
        # Compute filters coefficients for all thetas
        for idx, theta in enumerate(theta_vector):
            # Absolute value is taken to prevent R from going below zero. In Kuttruff, "Acoustics - an Introduction" the modulus of R is used
            # to compute reflections, and with our procedure we are not computing the imaginary part (see Nijl et al. Absorbing surfaces... 2006), 
            # so it is coherent.
            R = np.abs((Z * np.cos(math.radians(theta)) - Z0) / (Z * np.cos(math.radians(theta)) + Z0))
            b_fir[idx] = scipy.signal.firwin2(ntaps, freqs / 4000, R)
        
        return b_fir
    
    def _compute_air_impedance(self):
        ## Computes air impedance given:
        #   T: temperature in Kelvin
        #   p: atmospheric pressure in atm
        #   c: speed of sound in air, at temperature T
        self.environment.pressure = self.environment.pressure * 101325 # Convert atm to Pascal
        R_spec = 287.058
        return self.environment.pressure / (R_spec * self.environment.temperature) * self.environment.c

    def compute_asphalt_reflection_filter(theta, b_fir, theta_vector):
        idx = np.where(theta_vector == np.round(theta))
        idx = idx[0][0]
        return b_fir[idx]
    

    # Computes distance between source and receiver and delay in seconds.
    # Arguments are source position, receiver position and speed of sound in air
    def _compute_delay(self, src_pos, mic_pos, c):
        d = np.sqrt(np.sum((src_pos - mic_pos)) ** 2)
        tau = d / c
        return d, tau
    
    # Compute incidence angle of sound wave on road surface, given source and microphone position
    def _compute_angle(self, src_pos, mic_pos):
        # Distance between image and microphone
        dist = np.sqrt(np.sum(((src_pos - np.array([0, 0 , 2*src_pos[2]]) - mic_pos))) ** 2)
        # Incidence angle
        theta = np.arcsin(dist / (src_pos[2] + mic_pos[2]))
        return theta


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