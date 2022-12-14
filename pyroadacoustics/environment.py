import numpy as np
import math
import matplotlib.pyplot as plt

from .material import Material
from .soundSource import SoundSource
from .microphoneArray import MicrophoneArray
from .simulatorManager import SimulatorManager

class Environment:
    """
    This is the main class of the package and defines the Acoustic Scene that will be simulated. The goal of the
    package is the simulation of a moving sound source emitting sound above a flat surface. The sound propagation
    is simulated according to a model of the atmospheric sound propagation, and takes into account the sound
    that is being reflected by the road surface.

    The acoustic scene is composed of:
    * One sound source, emitting an arbitrary signal. It can be static or move along an arbitrary trajectory
    * One Microphone Array, containing a set of N omnidirectional microphones with an arbitrary disposition. The 
    array is static and is used to capture the sound produced by the moving source.
    * A set of environmental parameters describing the atmospheric conditions:
            * Atmospheric Pressure
            * Atmospheric Temperature
            * Relative Humidity
    * The material of the road surface
    * An optional diffuse background noise signal to be added to the simulation

    Alongside these parameters, simulation attributes are also defined:
    * The sampling frequency
    * The air absorption coefficients
    * The asphalt reflection coefficients

    From this class the simulation can be set up and run, via the instantiation of a SimulatorManager that
    contains the core functions of the simulator.

    Attributes
    ----------
    fs: int
        Sampling Frequency to be used for the simulations
    c: float
        Speed of sound in the air, computed from atmospheric parameters
    source: SoundSource
        Object containing parameters of the sound source
    background_noise: np.ndarray
        1D Array containing the samples of the background noise signal. The 
        background noise is assumed to be of diffuse type
    mic_array: MicrophoneArray
        Object containing the parameters of the microphone array (number and position of microphones)
    temperature: float
        Atmospheric temperature expressed in `Celsius` degrees
    pressure: float
        Atmospheric pressure expressed in `atm`
    rel_humidity: int
        Relative Humidity expressed as a percentage. Takes values between 0 and 100
    road_material: Material
        Object containing information about the absorption and reflection properties of the road surface
    air_absorption_coefficients: np.ndarray
        1D Array containing the air absorption coefficients computed at a set of 50 equispaced frequencies
        between 0 and `fs/2`
    Z0: float
        Value for air impedance computed at temperature `temperature`, pressure `pressure` and with 
        corresponding speed of sound `c`

    Methods
    -------
    set_road_material(absorption):
        Assign to the scene the material of the road surface, by selecting it from a database or creating a new
        material based on its acoustic absorption properties
    set_simulation_params(interp_method, include_reflection, include_air_absorption):
        Sets the parameters to be used in the simulation:
        * `interp_method`: interpolation method used to perform interpolated reads from delay line
        * `include_reflection`: if `False` only direct sound is simulated, if `True` also reflected path is included
        * `include_air_absorption`: if `True` effect of air absorption is included, if `False` it is neglected
    add_source(position, signal = None, trajectory = None):
        Creates a `SoundSource` by defining its initial position, its signal and (if it is a moving source) its
        trajectory, and inserts it into the acoustic scene
    set_background_noise(signal, SNR):
        Generates a background noise signal to be added to the signals generated at the microphone positions during
        the simulation. The noise level is defined by setting its SNR with respect to the source signal received by
        the microphone after the simulation is complete
    add_microphone_array(mic_locs):
        Creates a `MicrophoneArray` object by defining the positions of its microphones, and inserts it into the
        acoustic scene
    plot_environment():
        Plots the position of the microphones in the `MicrophoneArray` object and the trajectory of the `SoundSource`
    simulate():
        Runs the simulation and retrieves the signal received at each microphone in the `MicrophoneArray`

    """

    def __init__(
            self,
            fs: int = 8000,
            temperature: float = 20,
            pressure: float = 1,
            rel_humidity: int = 50,
            road_material: Material = Material('average_asphalt'),

        ) -> None:
        """
        Creates an Environment object by setting the simulation scene parameters. The Environment is defined by
        * a `SoundSource` object, representing a sound source moving along a trajectory
        * a `MicrophoneArray` object, representing a set of N microphones in fixed positions
        * a set of parameters describing the atmospheric conditions:
                * Temperature
                * Pressure
                * Relative Humidity
        * a `Material` object describing the absorption and reflection properties of the road surface
        * a sampling frequency `fs` used in the simulations

        Parameters
        ----------
        fs : int, optional
            Sampling frequency used in the simulations, by default 8000
        temperature : float, optional
            Atmospheric temperature in `Celsius` degrees, by default 20 degrees
        pressure : float, optional
            Atmospheric pressure in `atm`, by default 1 atm
        rel_humidity : int, optional
            Relative Humidity expressed as a percentage. Takes values in [0, 100], by default 50
        road_material : Material, optional
            Material object containing absorption and reflection properties of road surface, by default None. If 
            is None, an average asphalt material is chosen from Database.
        
        Raises
        ------
        ValueError:
            if rel_humidity is lower than zero or greater than 100
        """

        self.fs = fs
        self.source = None
        self.mic_array = None
        self.temperature = temperature
        self.pressure = pressure
        if rel_humidity < 0 or rel_humidity > 100:
            raise ValueError("Humidity must be greater than zero and lower than 100")
        self.rel_humidity = rel_humidity
        
        if road_material.absorption["center_freqs"][-1] != self.fs / 2:
            road_material.extrapolate_coeffs_to_spectrum(fs = self.fs)
        self.road_material = road_material

        simulation_params: dict = {
                "interp_method": "Sinc",
                "include_reflected_path": True,
                "include_air_absorption": True,
        }
        self.simulation_params = simulation_params
        
        self.c = self._compute_speed_sound(temperature)
        self.air_absorption_coefficients = self._compute_air_absorption_coefficients()
        self.Z0 = self._compute_air_impedance(self.temperature, self.pressure, self.c)

        # If true, background noise will be introduced
        self._background_noise_flag = False
        self._background_noise_SNR = 0
        self._background_noise = None
    
    def set_road_material(self, absorption: str or dict) -> None:
        """
        Set the road material by creating a new Material object from the absorption parameter. This parameter
        can either be a str containing a material name(the material will be chosen from the database based on 
        its name) or a dict containing keys `"coeffs"` and `"freqs"` to define the absorption properties of 
        an arbitrary material. See documentation of class `Material` for more information.

        Parameters
        ----------
        absorption : str or dict
            * If `str` is given: chooses `Material` from database
            * If `dict` is given: defines new `Material` having the `"coeffs"` and `"freqs"` specified in the
            dictionary.

        Modifies
        --------
        material
            The chosen or created material is assigned to the corresponding `Environment` attribute
        """

        material = Material(absorption)
        
        # If frequencies do not cover the whole spectrum, perform extrapolation
        if material.absorption["center_freqs"][-1] != self.fs / 2:
            material.extrapolate_coeffs_to_spectrum(fs = self.fs)
        
        self.road_material = material
    
    def set_simulation_params(self, interp_method: str = 'Allpass', include_reflection: bool = True, 
        include_air_absorption: bool = True) -> None:
        """
        Function used to define a set of parameters to be used in the simulation. These parameters are:
        * The interpolation method used for the interpolated reads from the delay lines. `Sinc` provides best
        accuracy but has highest computational load.
        * The possibility to include the reflected sound path in the simulation. If it is not included, the source is
        assumed to emit sound in the free-field and just the direct path is simulated: this reduces the computational
        load but also the physical accuracy
        * The possibility to include the air absorption in the simulation. The air absorption depends on distance and is
        implemented as a Time Varying FIR filter. The computation of its coefficients is performed at each simulation
        instant, and implies an increase of the computational complexity. This can be neglected, leading to a reduction in the
        complexity but also in the simulation accuracy.

        Parameters
        ----------
        interp_method : str
            Interpolation method used to perform interpolated reads from the delay lines. Can be:
            * `Linear`: linear interpolation
            * `Allpass` (default): first order allpass interpolation
            * `Sinc`: sinc interpolation using a windowed sinc filter with 11 taps. Highest accuracy and
            computational complexity 
        include_reflection : bool
            Bool, if `True` (default) the simulation includes the direct sound path and the road
            surface reflection. If `False` just the direct sound is simulated.
        include_air_absorption : bool
            Bool, if `True` (default), the air absorption is included in the simulation. The air absorption is modeled 
            using a FIR filter with 11 taps, whose coefficients depend on the distance between source and receiver 
            and thus are updated at each simulation instant. To reduce the computational load, this can be switched off 
            by setting this to `False`.
        """
        simulation_params = {
                "interp_method": interp_method,
                "include_reflected_path": include_reflection,
                "include_air_absorption": include_air_absorption,
            }
        self.simulation_params = simulation_params
    
    def add_source(self, position: np.ndarray, signal: np.ndarray = None, 
        trajectory_points: np.ndarray = None, source_velocity: np.ndarray or float = None) -> None:
        """
        Creates a sound source and adds it to the environment. To create the `SoundSource` object, 
        the specified parameters are required

        Parameters
        ----------
        position : np.ndarray
            Initial source position, specified as a 1D array containing three cartesian coordinates [x,y,z]
        signal : np.ndarray, optional
            Signal emitted by the sound source, by default None. If the signal is None, a default frequency 
            modulated sinusoidal signal is assigned to the source. The signal is looped to cover the whole 
            duration of the simulation
        trajectory_points : np.ndarray, optional
            2D Array containing N sets of 3 cartesian coordinates `[x,y,z]` defining the desired trajectory positions.
            Each couple of subsequent points define a straight segment on the overall trajectory. If value is None,
            the source is assumed to be static
        source_velocity : np.ndarray or float, optional
            * 2D Array containing N-1 floats defining the modulus of the velocity on each trajectory segment
            * float defining the modulus of the velocity on the whole trajectory (i.e. constant speed)

            If source is moving, it must be assigned. If source is static, it takes default value `None`

        Raises
        ------
        RuntimeError:
            If a source is already present in the acoustic scene (self.source != None)
        RuntimeError:
            If the trajectory or position of the source contain the height z = 0
        
        Modifies
        --------
        source
            The created `SoundSource` object is assigned to the corresponding attribute
        """
        #TODO: Normalize source signal
        if self.source is not None:
            raise RuntimeError("Cannot insert more than one sound source")
        is_static = False
        if position[2] <=1e-5 or trajectory_points[:,2].any() <=1e-5:
            raise RuntimeError('The source should always have a height greater than 0')
        if trajectory_points is None:
            is_static = True

            if signal is not None:
                source = SoundSource(position,self.fs,is_static,static_simduration=len(signal)/self.fs)
            else:
                source = SoundSource(position, self.fs, is_static)
        else:
            source = SoundSource(position = position, fs = self.fs, is_static = False)
            source.set_trajectory(trajectory_points, source_velocity)

        if signal is None:
            # Define a default signal --> sinusoidal siren
            simulation_duration = len(source.trajectory) / self.fs
            
            ## Define sinusoid with frequency modulation
            f = 800
            f_lfo = 0.3

            t = np.arange(0, simulation_duration, 1/self.fs)
            signal = np.zeros_like(t)
            for i in range(len(t)):
                signal[i] = 0.2 * np.sin(2 * np.pi * f * t[i] + 600 * np.sin(2 * np.pi * f_lfo * t[i]))

        # If trajectory is longer than signal, loop the signal
        while len(signal) < len(source.trajectory):
            signal = np.append(signal, signal)
        if len(signal) > len(source.trajectory):
            signal = signal[0:len(source.trajectory)]
        
        # Add signal to the sound source
        source.set_signal(signal)

        self.source = source
    
    def _add_noise_source(self, position: np.ndarray, signal: np.ndarray = None) -> None:
        """
        Creates a static noise signal and adds it to the acoustic scene in a specified position. Still
        to be implemented.

        Parameters
        ----------
        position : np.ndarray
            Noise source position, specified as a 1D array containing three cartesian coordinates [x,y,z]
        signal : [type], optional
            Signal emitted by the sound source, by default None. If the signal is None, a default white  
            noise signal is assigned to the source. The signal is looped to cover the whole  duration of 
            the simulation

        Raises
        ------
        NotImplementedError
            Function is still to be implemented
        """

        raise NotImplementedError('Noise Sources are not supported yet')
    
    def set_background_noise(self, signal: np.ndarray = None, SNR: float = 15) -> None:
        """
        Enables the use of a static background noise in the simulation. The noise is assumed to be of 
        diffuse type, and is defined by its `SNR`. This is computed as the SNR between the source signal
        received by the microphone after the simulation is complete, and the noise signal chosen by the 
        user. If no noise signal is chosen, an static white noise is used.

        Parameters
        ----------
        signal : np.ndarray, optional
            1D ndarray containing the samples of the background noise signal to be used during the simulation, 
            by default None
        SNR : float, optional
            Signal to Noise ratio in dB scale, by default 15

        Raises
        ------
        RuntimeError
            If the source or the mic_array has not been instantiated yet
        """

        if self.source == None or self.mic_array == None:
            raise RuntimeError("To add a background noise you need to first insert a sound source"
                "and a microphone array")

        self._background_noise_flag = True
        self._background_noise_SNR = SNR
        
        if signal is None:
            # Define default signal --> white noise
            simulation_duration = len(self.source.trajectory) / self.fs
            t = np.arange(0, simulation_duration, 1 / self.fs)
            signal = np.random.randn(len(t))
        
        else:
            while(len(signal) < len(self.source.trajectory)):
                signal = np.append(signal, signal)
            if len(signal) > len(self.source.trajectory):
                signal = signal[0:len(self.source.trajectory)]
        
        self._background_noise = signal

    def add_microphone_array(self, mic_locs: np.ndarray) -> None:
        """
        Creates a MicrophoneArray object and adds it to the Environment. The array is defined by the 
        position of its microphones.

        Parameters
        ----------
        mic_locs : np.ndarray
            2D Array containing N triplets of cartesian coordinates [x,y,z] defining the position of the 
            microphones of the array
        
        Raises
        ------
        RuntimeError:
            If a microphone array is already present in the acoustic scene when method is called

        Modifies
        --------
        mic_array
            The instantiated MicrophoneArray is assigned to the corresponding Environment attribute
        """
        if self.mic_array != None:
            raise RuntimeError("Cannot insert more than one microphone array")
        mic_array = MicrophoneArray(mic_locs)
        self.mic_array = mic_array

    def plot_environment(self):
        """
        Plots a 2D representation of the position of the microphones in `mic_array` and the trajectory of the 
        sound source during the simulation.
        """

        plt.figure()
        plt.plot(self.source.trajectory[:,0], self.source.trajectory[:,1])
        plt.scatter(self.mic_array.mic_positions[:,0], self.mic_array.mic_positions[:,1], color = 'green', marker='x')
        plt.legend(['Source Trajectory', 'Microphones'])
        plt.xlabel('[m]')
        plt.ylabel('[m]')
        plt.suptitle('Simulation Scenario')
        plt.title('Source and Microphones have height: z = %.2f' %self.mic_array.mic_positions[0,2])
        plt.show()

    def simulate(self) -> np.ndarray:
        """
        Runs the simulation and returns received microphone signals. To be called, the `Environment` must be fully
        specified, i.e. it must contain a `SoundSource` and a `MicrophoneArray`. The function relies on the
        `SimulatorManager`, a class that contains the functions to compute the new
        simulator output samples (i.e. the signals received at the microphones) and to update the acoustic scene
        at each simulation frame.

        Returns
        -------
        np.ndarray
            2D Array containing N samples of the signals received by each of the M microphones in the defined 
            microphone array.

        Raises
        ------
        RuntimeError
            If Source or MicrophoneArray have not been defined yet
        """
        
        if self.source == None or self.mic_array == None:
            raise RuntimeError('Before running simulation, you must define a sound source and a microphone array')
        
        # Duration of the simulation
        N = len(self.source.trajectory)
        M = self.mic_array.nmics

        signals = np.zeros((M,N))

        for m in range(M):
            # Select Active Microphone
            active_mic = self.mic_array.mic_positions[m]
        
            # Instantiate Simulator Manager
            manager = SimulatorManager(c = self.c, fs = self.fs, Z0 = self.Z0, road_material = self.road_material,
                airAbsorptionCoefficients = self.air_absorption_coefficients, simulation_params = self.simulation_params)

            # Define simulation loop
            manager.initialize(self.source.position, active_mic)

            # Compute output samples
            for n in range(N):
                signals[m,n] = manager.update(self.source.trajectory[n], active_mic, self.source.signal[n])
            

            # Add background noise
            if self._background_noise_flag == True:
                # Compute attenuation factor from SNR
                noise_attenuation = np.sqrt(np.sum(signals[m]** 2) / (10 ** (self._background_noise_SNR/10) 
                    * np.sum(self._background_noise ** 2)))
                
                # Compute background noise signal
                noise = self._background_noise * noise_attenuation
                
                # Compute signal
                signals[m] = signals[m] + noise
        return signals
    
    def _compute_air_absorption_coefficients(self, nbands: int = 20) -> np.ndarray:
        """
        Computes air absorption coefficients at a set of nbands equispaced frequencies in the range `[0, fs]`, based
        on the ISO 9613-1 standard. The coefficients depend on the atmsopsheric temperature, pressure and relative humidity.

        Parameters
        ----------
        nbands: int
            Number of frequency bands in which to compute air absorption coefficients, by default 20

        Returns
        -------
        np.ndarray
            1D Array containing nbands air absorption coefficients, computed in dB scale, at the defined 
            set of frequencies

        """

        T0 = 293.15     # Standard room temperature T = 20deg Celsius
        T01 = 273.16    # Triple point isotherm temperature, ISO 9613-1
        T = self.temperature + 273.15
        ps0 = 1         # Standard atmospheric pressure in atm

        f = np.linspace(0, self.fs / 2, num=nbands)     # Frequencies in which coeffs will be computed
        
        Csat = -6.8346 * math.pow(T01 / T, 1.261) + 4.6151
        rhosat = math.pow(10, Csat)
        H = rhosat * self.rel_humidity * ps0 / self.pressure

        frn = (self.pressure / ps0) * math.pow(T0 / T, 0.5) * (
                9 + 280 * H * math.exp(-4.17 * (math.pow(T0 / T, 1/3.) - 1)))

        fro = (self.pressure / ps0) * (24.0 + 4.04e4 * H * (0.02 + H) / (0.391 + H))

        alpha = f * f * (
            1.84e-11 / ( math.pow(T0 / T, 0.5) * self.pressure / ps0 )
            + math.pow(T / T0, -2.5)
            * (
                0.10680 * math.exp(-3352 / T) * frn / (f * f + frn * frn)
                + 0.01278 * math.exp(-2239.1 / T) * fro / (f * f + fro * fro)
                )
            )
        
        return alpha * 20 / np.log(10)
    
    def _compute_air_impedance(self, T: float, p: float = 1, c: float = None) -> float:
        """
        Compute Specific impedance of air, given temperature (in Celsius) and pressure (in atm). Speed of 
        sound can be given as a parameter; if None it is computed from temperature parameter.

        Parameters
        ----------
        T : float
            Atmospheric temperature in `Celsius` degrees
        p : float, optional
            Atmospheric pressure in `atm`, by default 1 atm
        c : float, optional
            Speed of sound in the air at T temperature, by default None. It can be given to speed up computation,
            otherwise it is computed from T inside the method

        Returns
        -------
        float
            Specific impedance Z0 of air

        """
        
        if c == None:
            c = self._compute_speed_sound(T)
        
        p = p * 101325          # Convert atm to Pascal
        T = T + 273.15          # Convert temperature to Kelvin
        R_spec = 287.058
        
        return p / (R_spec * T) * c

    def _compute_speed_sound(self, T: float) -> float:
        """
        Computes the speed of sound in the air at temperature T (expressed in Celsius degrees)

        Parameters
        ----------
        T : float
            Atmospheric Temperature expressed in Celsius degrees

        Returns
        -------
        float
            Value of speed of sound in air at temperature T

        """

        return 331.3 * np.sqrt(1 + T / 273.15)
