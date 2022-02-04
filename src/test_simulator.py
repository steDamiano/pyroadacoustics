import simulator_manager as sim
import source as src
import microphone_array
import numpy as np

source = src.Source(velocity = 5)
P1 = np.array([3,10])
P2 = np.array([3, -5])
P3 = np.array([[0,0]])

source.set_trajectory(P1, P2)

fs = 8000
f = 400
time = np.arange(0, 6, 1/fs)

# Sinusoidal
# signal = np.sin(2 * np.pi * f * time)

# Noise
signal = np.random.randn(len(time))

# Train impulse
# signal = np.zeros_like(time)
# signal[::100] = 1


mic = microphone_array.MicrophoneArray(P3)
manager = sim.SimulatorManager(source, mic, fs)

manager.run_simulation()
