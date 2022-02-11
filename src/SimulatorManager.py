import numpy as np
#from Environment import Environment
from DelayLine import DelayLine

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
            # Variable contains c and fs
            environment: np.ndarray,
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
        self.primaryDelLine.set_delays(np.array([tau * self.environment.fs, tau_1 * self.environment.fs]))
        self.secondaryDelLine.set_delays(tau_2 * self.environment.fs)
        

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
        y_primary = self.primaryDelLine.update_delay_line(signal_sample, np.array([tau * self.environment.fs, 
            tau_1 * self.environment.fs]))

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
        y_secondary = self.primaryDelLine.update_delay_line(sample_eval, np.array([tau_2 * self.environment.fs]))
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