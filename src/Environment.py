import numpy as np
import math
import matplotlib.pyplot as plt

from src.Material import Material
from src.SoundSource import SoundSource
from src.MicrophoneArray import MicrophoneArray
from src.SimulatorManager import SimulatorManager

class Environment:
    """
    This class is the main class of the package and defines the Acoustic Scene that will be simulated. The 
    package aims at the simulation of sound source emitting sound above a flat surface. The sound propagation
    is simulated according to a model of the atmospheric sound propagation and takes into account the sound
    that is being reflected by the asphalt surface of the road.

    The acoustic scene is composed of:
    * One sound source, that is emitting an arbitrary signal and can be stati or move on an arbitrary trajectory
    * One Microphone Array, containing a set of N omnidirectionalmicrophones with an arbitrary disposition. The 
    array is static and is used to capture the sound produced by the moving source.
    * A set of environmental parameters describing the atmospheric model:
            * Atmospheric Pressure
            * Atmospheric Temperature
            * Relative Humidity
    * The material of the road surface
    * An optional diffuse background noise signal that is present in the environment

    Alongside this parameters, simulation attributes are also defined:
    * The sampling frequency to be used
    * The air absorption coefficients
    * TODO the asphalt reflection coefficients

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
        1D Array containing the samples of the background noise signal present in the scene. The 
        background noise is assumed to be of diffuse type
    mic_array: MicrophoneArray
        Object containing the parameters of the microphone array (number and position of microphones)
    temperature: float
        Atmospheric temperature expressed in `Celsius` degrees
    pressure: float
        Atmospheric pressure expressed in `atm`
    rel_humidity: int
        Relative Humidity expressed as a percentage. Takes value between 0 and 100
    road_material: Material
        Object containing information about the absorption and reflection properties of the road surface
    air_absorption_coefficients: np.ndarray
        1D Array containing the air absorption coefficients computed at a set of 50 equispaced frequencies
        between 0 and `fs/2`

    Methods
    -------
    get_road_material(self, absorption):
        Assign to the scene the material of the road surface, by selecting it from a database or creating a new
        material based on its acoustic absorption properties
    add_source(self, position, signal = None, trajectory = None):
        Creates a `SoundSource` by defining its initial position, its signal and (if it is a moving source) its
        trajectory, and inserts it into the acoustic scene
    add_background_noise(self, signal, SNR):
        Generates a background noise signal to be added to the signals generated at the microphone positions during
        the simulation. The noise level is defined by setting its SNR with respect to the source signal when the 
        source is closest to the microphone array, while travelling on its trajectory
    add_microphone_array(self, mic_locs):
        Creates a `MicrophoneArray` object by defining the positions of its microphones, and inserts it into the
        acoustic scene
    plot_environment(self):
        Plots the position of the microphones in the `MicrophoneArray` object and the trajectory of the `SoundSource`
    simulate(self):
        Runs the simulation and retrieves the signal received at each microphone in the `MicrophoneArray`

    """
    def __init__(
            self,
            fs: int = 8000,
            source: SoundSource = None,
            background_noise: np.ndarray = None,
            mic_array: MicrophoneArray = None,
            temperature: float = 20,
            pressure: float = 1,
            rel_humidity: int = 50,
            road_material: Material = Material('average_asphalt')
        ) -> None:
        """
        Creates an Environment object by setting the simulation scene parameters. The Environment is defined by
        * a `SoundSource` object, representing a sound source moving along a trajectory
        * a `MicrophoneArray` object, representing a set of N microphones in certain positions in the acoustic scene
        * a set of parameters describing the atmospheric conditions:
                * Temperature
                * Pressure
                * Relative Humidity
        * a `Material` object describing the absorption and reflection properties of the road surface
        * a sampling frequency `fs` using in the simulations

        Parameters
        ----------
        fs : int, optional
            Sampling frequency used in the simulations, by default 8000
        source : SoundSource, optional
            Sound source that produces simulated sound, by default None
        background_noise : np.ndarray, optional
            1D array that contains samples of the background noise to be added in the simulations, by default None.
            The noise is assumed to be of type diffuse
        mic_array : MicrophoneArray, optional
            Microphone array containing positions of microphones that record sound in the scene, by default None
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
        self.source = source
        self.background_noise = background_noise
        self.mic_array = mic_array
        self.temperature = temperature
        self.pressure = pressure
        if rel_humidity < 0 or rel_humidity > 100:
            raise ValueError("Humidity must be greater than zero and lower than 100")
        self.rel_humidity = rel_humidity
        self.road_material = road_material
        
        self.c = self._compute_speed_sound(temperature)
        self.air_absorption_coefficients = self._compute_air_absorption_coefficients()
    
    def set_road_material(self, absorption: str or dict) -> None:
        """
        Set the road material by creating a new Material object from the absorption parameter. This parameter
        can either be a str containing a material name(the material will be chosen from the database based on 
        its name) or a dict containing keys `"coeffs"` and `"freqs"` to define the absorption properties of 
        an arbitrary material. See documentation of class `Material` for more information.

        Parameters
        ----------
        absorption : str or dict
            * If `str` is given: choose `Material` from database
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

            If source is moving, it must be assigned. If source is static, it takes default value 1

        Raises
        ------
        RuntimeError:
            If a source is already present in the acoustic scene (self.source != None)
        
        Modifies
        --------
        source
            The created `SoundSource` object is assigned to the corresponding attribute
        """
        if self.source != None:
            raise RuntimeError("Cannot insert more than one sound source")
        is_static = False
        if type(trajectory_points) != np.ndarray:
            is_static = True
            # Define duration of the simulation
            # simulation_duration = 5
            # trajectory = np.tile(position, (simulation_duration * self.fs))
            source = SoundSource(position, self.fs, is_static)
        else:
            source = SoundSource(position = position, fs = self.fs, is_static = False)
            source.set_trajectory(trajectory_points, source_velocity)

        if signal == None:
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
        while len(source.trajectory) < len(signal):
            np.append(signal, signal)

        self.source = source
    
    def add_noise_source(self, position: np.ndarray, signal: np.ndarray = None) -> None:
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

    def add_background_noise(self, signal: np.ndarray = None, SNR: float = 15) -> None:
        """
        Defines background noise signal to be added to the simulation. The background noise is assumed
        to be of diffuse type, and is defined based on the SNR in dB scalebetween the noise signal and source 
        signal, computed in the position along the trajectory closest to the first microphone of the microphone 
        array.

        Parameters
        ----------
        signal : np.ndarray
            1D ndarray containing the samples of the background noise signal to be used during the simulation
        SNR : float
            Signal to Noise ratio in dB scale. The SNR is defined as the ratio between the source signal and the
            noise signal, computed using the source signal received by the first microphone of the array when the
            source is closest to it while travelling along its trajectory. Default SNR is 15 dB

        Raises
        ------
        RuntimeError
            If the source has not been instantiated yet
        
        Modifies
        --------
        background_noise:
            The computed background noise signal samples are assigned to the corresponding attribute of the class

        """
        if self.source == None:
            raise RuntimeError("To add a background noise you need to first insert a sound source")
        
        if signal == None:
            # Define default signal --> white noise
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
        noise_attenuation = np.sqrt(np.sum((self.source.signal ** 2) / (4 * np.pi * d_min)) / 
            (10 ** (SNR/10) * np.sum((signal ** 2) / (4 * np.pi * d_min))))
        signal = signal * noise_attenuation
        
        self.background_noise = signal


    
    def add_microphone_array(self, mic_locs: np.ndarray) -> None:
        """
        Creates a MicrophoneArray object and atts it to the Environment. The array is defined by the 
        position of its microphones.

        Parameters
        ----------
        mic_locs : np.ndarray
            2D Array containing N triplets of cartesian coordinates [x,y,z] defining the position of the 
            microphones of the array

        Modifies
        --------
        mic_array
            The instantiated MicrophoneArray is assigned to the corresponding Environment attribute
        """

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

    # Runs simulation. Returns array np.array([M,N]), where M is the number of microphones 
    # and N is the number of samples of the simulation. Array contains signals recorded by microphones
    def simulate(self) -> np.ndarray:
        """
        Runs the simulation and returns received microphone signals. To be called, the `Environment` must be fully
        specified, i.e. it must contain a `SoundSource` and a `MicrophoneArray`. The function relies on the
        `SimulatorManager`, an instance of a singleton class that contains the functions to compute the new
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
        
            # Instantiate Simulator Manager or call instance
            manager = SimulatorManager(environment = self, active_microphone = active_mic, source = self.source, 
            airAbsorptionFilters = self.air_absorption_coefficients)

            # Define simulation loop
            _temp = manager.initialize(self.source.position, active_mic)

            # Compute output samples
            for n in range(N):
                signals[m,n] = manager.update(self.source.trajectory[n], active_mic, self.source.signal[n])
    
    def _compute_air_absorption_coefficients(self, nbands: int = 50) -> np.ndarray:
        """
        Computes air absorption coefficients at a set of nbands equispaced frequencies in the range `[0, fs]`, based
        on the ISO 9613-1 standard. 

        A different formulation can be found in 
        `Keith Attenborough, "Sound Propagation in the Atmosphere", Springer Handbook of Acoustics`, and can be set
        by using T01 = 293.15
        The coefficients depend on the atmsopsheric temperature, pressure and relative humidity.

        Parameters
        ----------
        nbands: int
            Number of frequency bands in which to compute air absorption coefficients, by default 50

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

        f = np.linspace(0, self.fs, num=nbands)     # Frequencies in which coeffs will be computed
        
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
        sound can be given as a parameter, if None it is computed from temperature parameter.

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
