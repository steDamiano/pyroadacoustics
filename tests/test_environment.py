from multiprocessing.sharedctypes import Value
from turtle import position
import matplotlib.pyplot as plt
import numpy as np
import unittest
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Material import Material
from src.SoundSource import SoundSource
from src.Environment import Environment

class EnvironmentTest(unittest.TestCase):
    def test_instantiate(self):
        # Normal instantiation
        env = Environment()
        self.assertTrue(env.road_material.absorption["description"], "Average asphalt model, see Notebook for details")
        self.assertEqual(env.source, None)
        self.assertEqual(env.mic_array, None)
        self.assertEqual(env.fs, 8000)
        self.assertEqual(env.background_noise, None)
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
        # Ground Truth with Springer model
        # alpha_expected = np.array([0., 0.0130, 0.0441, 0.0839, 0.1237, 0.1589, 0.1886, 0.2131])

        # Ground Truth with ISO model
        alpha_expected = np.array([0., 0.00525, 0.01195, 0.02276, 0.03770, 0.05659, 0.07927, 0.10550])
        self.assertTrue(np.allclose(alpha, alpha_expected, rtol=0.001))
    
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
        print(env.source)
        env.add_source(position = np.array([0., 3., 5.]), 
            trajectory_points = np.array([[0., 3., 5.], [0., 3., -5.]]), source_velocity = np.array([5]))
        self.assertEqual(env.source.is_static, False)


if __name__ == '__main__':
    unittest.main()