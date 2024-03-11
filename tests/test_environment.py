import matplotlib.pyplot as plt
import numpy as np
import unittest
import os, sys
import scipy.signal
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyroadacoustics.material import Material
from pyroadacoustics.environment import Environment

class EnvironmentTest(unittest.TestCase):
    def test_instantiate(self):
        # Normal instantiation
        env = Environment(road_material=Material('average_asphalt'))
        self.assertTrue(env.road_material.absorption["description"], "Average asphalt model, see Notebook for details")
        self.assertEqual(env.source, None)
        self.assertEqual(env.mic_array, None)
        self.assertEqual(env.fs, 8000)
        self.assertEqual(env._background_noise, None)
        self.assertEqual(env.temperature, 20)
        self.assertEqual(env.pressure, 1)
        self.assertEqual(env.rel_humidity, 50)

        # Wrong values for Humidity
        self.assertRaises(ValueError, Environment, rel_humidity = 120)
        self.assertRaises(ValueError, Environment, rel_humidity = -1)
    
    def test_compute_soundspeed(self):
        env = Environment(temperature=20)
        self.assertAlmostEqual(env.c, 343.2146, 4)
        env = Environment(temperature=25)
        self.assertAlmostEqual(env.c, 346.1292, 4)

    def test_compute_air_impedance(self):
        env = Environment(temperature=20, pressure=1)
        self.assertAlmostEqual(env._compute_air_impedance(env.temperature, env.pressure, env.c), 413.2595, 4)
        env = Environment(temperature=25, pressure=1.35)
        self.assertAlmostEqual(env._compute_air_impedance(env.temperature, env.pressure, env.c), 553.2025, 4)
    
    def test_compute_airabs_coeffs(self):
        env = Environment(temperature=20, pressure=1, rel_humidity=50)
        alpha = env._compute_air_absorption_coefficients(nbands=8)

        # Ground Truth with ISO model
        alpha_expected = np.array([0, 0.003035195287164, 0.005249959372774, 0.008106658065295, 0.011952756167730,
            0.016837651366320, 0.022763968109445, 0.029721153743089])
        self.assertTrue(np.allclose(alpha, alpha_expected))
    
    def test_set_road_material(self):
        env = Environment()
        # Set road material from database
        env.set_road_material('m4_asphalt')
        self.assertEqual(env.road_material.absorption["description"], "M4 Asphalt Mixture")

        # Set road material from dictionary
        custom_mat = {
            "description": "Custom Material",
            "coeffs": [0., 0.01, 0.02, 0.03],
            "center_freqs": [100, 1000, 3000, 4000]
        }
        env.set_road_material(custom_mat)
        self.assertEqual(env.road_material.absorption["description"], "Custom Material")

        # Set road material at instantiation
        mat = Material(custom_mat)
        env2 = Environment(road_material=mat)
        self.assertEqual(env2.road_material.absorption["description"], "Custom Material")

    def test_add_source(self):
        # Add source static
        env = Environment()
        env.add_source(position=np.array([0., 3., 5.]))
        self.assertTrue(np.all(env.source.position == np.array([0., 3., 5.])))
        self.assertTrue(env.source.is_static)

        # Add second source
        self.assertRaises(RuntimeError, env.add_source, position = np.array([0., 0., 0.]))

        # Add a dynamic source
        env = Environment()
        env.add_source(position = np.array([0., 3., 5.]), 
            trajectory_points = np.array([[0., 3., 5.], [0., 3., -5.]]), source_velocity = np.array([5]))
        self.assertEqual(env.source.is_static, False)
    
    def test_add_micarray(self):
        # Add wrong elements
        env = Environment()
        self.assertRaises(ValueError, env.add_microphone_array, np.array([[0,0,0], [0,1]], dtype=object))

        # Add single microphone
        env = Environment()
        env.add_microphone_array(np.array([[1,3,2]]))
        self.assertEqual(env.mic_array.nmics, 1)

        # Add second array
        self.assertRaises(RuntimeError, env.add_microphone_array, np.array([0,0,0]))

        # Add multiple microphones
        env = Environment()
        positions = np.array([[1, 1, 1], [1, 1.2, 1], [1, 1.4, 1]])
        env.add_microphone_array(positions)
        self.assertEqual(env.mic_array.nmics, 3)
        self.assertTrue(np.array_equiv(env.mic_array.mic_positions, positions))
    
    def test_background_noise(self):
        noise_sig = np.random.randn(3000) # np.random.randn(3000)

        # Add noise without source
        env = Environment()
        self.assertRaises(RuntimeError, env.set_background_noise, noise_sig)
        env.add_microphone_array(np.array([[0,0,0]]))
        self.assertRaises(RuntimeError, env.set_background_noise, noise_sig)

        # Add noise without microphones
        env = Environment()
        env.add_source(position=np.array([0,3,5]))
        self.assertRaises(RuntimeError, env.set_background_noise, noise_sig)

        # Add noise for static source scene
        env = Environment()
        src_signal = np.ones(1000)
        env.add_microphone_array(np.array([[0,0,0]]))
        env.add_source(position = np.array([0, 3, 1]), signal = src_signal)
        env.set_background_noise(noise_sig, SNR = 10)
        
        self.assertEqual(env._background_noise_SNR, 10)
        self.assertEqual(env._background_noise_flag, True)
        self.assertTrue(np.array_equiv(env._background_noise, noise_sig[0:len(src_signal)]))

    def test_plot(self):
        env = Environment()
        env.add_microphone_array(np.array([[0.,0.,0.],[0.,0.5,0.],[0.,1.,0]]))
        env.add_source(position = np.array([3, 5, 1]), trajectory_points = np.array([[3,5,1], [3,-5,1], [0,-5,1]]), 
            source_velocity = np.array([5, 2]))
        # env.plot_environment()
    
    def test_simulation(self):
        src_signal = np.random.randn(1000)
        env = Environment()
        env.add_source(position = np.array([3, 5, 1]), trajectory_points = np.array([[3,10,1], [3,-20,1]]), 
            source_velocity = np.array([5]), signal = src_signal)
        env.add_microphone_array(np.array([[0.,0.,1.]]))#,[0.,0.5,1.],[0.,1.,1.]]))
        env.set_simulation_params("Linear", True, True)
        signals = env.simulate()

        # # plot signals received at the 3 microphones
        # plt.figure()
        # fig, ax = plt.subplots(3,1)
        # ax[0].plot(signals[0,100:150])
        # ax[1].plot(signals[1,100:150])
        # ax[2].plot(signals[2,100:150])
        # plt.show()

        # plt.figure()
        # ff, tt, Sxx = scipy.signal.spectrogram(1 * signals[0,:], 8000)
        # plt.pcolormesh(tt, ff, Sxx, shading = 'auto')
        # plt.show()


if __name__ == '__main__':
    # import cProfile
    # cProfile.run('unittest.main()')
    unittest.main()
    # tst = EnvironmentTest()
    # tst.test_instantiate()