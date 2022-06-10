# pyroadacoustics
Pyroadacoustics is an open-source library that enables to simulate the sound propagation in a road scenario. The Python, entirely developed in Python, is built on a geometrical acoustics model to resemble the main acoustic phenomena that affect the sound produced by sources moving on a road, in a realistic outdoor environment.

The simulated acoustic scene is defined by:
- One omnidirectional sound source, moving on an arbitrary trajectory with an arbitrary speed, that emits a user-defined sound signal.
- A static array of omnidirectional microphones, with arbitrary geometry, used to record the sound produced by the source.
- A static background noise with defined SNR.
- A set of environmental parameters that define the atmospheric conditions:
    - Temperature
    - Pressure
    - Relative Humidity
- The material of the road surface over which the sound source is moving. This material characterizes the acoustic properties of the ground surface used in the simulation.

The simulator provides an accurate model of the *Doppler effect*, the *acoustic atmospheric absorption*, affecting sound propagation at high distances, and of the *asphalt reflection properties*.

## Installation and Use
To install and use *pyroadacoustics*, the following python packages are required:
- numpy
- scipy
- matplotlib

You can install the package using pip:

    pip install pyroadacoustics

As an alternative, you can download the repository to a local directory, and run

    python setup.py install


## Documentation and Examples
The documentation of all the classes and methods of the package is available in the corresponding docstring. 

An example of the definition of the simulation scene and of the use of the simulator is provided in the notebook `simulator_demo.ipynb`.

## Audio Demos
The directory `audio_demos` contains the audio files of  a set of simulations performed in a fixed scenario, with different source signals. The demos were produced with the `simulator_demo.ipynb` code. The scenario is defined by the following parameters:
- Atmospheric conditions: 
    - Temperature T = 20° C
    - Pressure P = 1 atm
    - Relative Humidity H = 50 %
- Sampling frequency: Fs = 8 kHz
- Static Microphone at position: [0,0,1] m
- Sound source moving on a straight trajectory starting at position [3,20,1] m and ending at position [3,-20,1] m, with constant speed v = 5m /s
- No background noise

The 4 demos were run with different sound source signals:

1. `demo1`: sinusoid with frequency F = 2000 Hz, without reflection from the ground (i.e. only direct sound field is simulated).
2. `demo2`: sinusoid with frequency F = 2000 Hz, with direct sound field and reflection from the ground.
3. `demo3`: white noise signal, with direct sound field and reflection from the ground.
4. `demo4`: European ambulance siren sound, with direct sound field and reflection from the ground.

## Acknowledgements
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 956962 (I-SPOT Project).
