import unittest
import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyroadacoustics.material import Material

class MaterialTest(unittest.TestCase):
    # Test object instantiation 
    def test_instantiate(self):
        # Wrong data type
        self.assertRaises(TypeError, Material, 2)
        
        # String in database
        material = Material('m1_asphalt').absorption
        self.assertIsInstance(material, dict)

        # String not in database
        self.assertRaises(KeyError, Material, 'adas')

        # Dictionary with missing key
        input = {
            "coeffs": [1,2],
            "foo": 'Prova'
        }
        self.assertRaises(KeyError, Material, input)

        input = {
            "center_freqs": [1,2],
            "foo": 'Prova'
        }
        self.assertRaises(KeyError, Material, input)

        # Dictionary with mismatched keys
        input = {
            "center_freqs": [0,1,2],
            "coeffs": [0,1]
        }
        self.assertRaises(ValueError, Material, input)
        

    def test_extrapolation(self):

        # Test number of bands
        material = Material('average_asphalt')
        coeffs = material.extrapolate_coeffs_to_spectrum(n_bands=50)
        self.assertEqual(len(coeffs), 50)
        # plt.figure()
        # plt.plot(material.absorption["center_freqs"], material.absorption["coeffs"])
        # plt.show()
        
        # Test interpolation method
        material = Material('average_asphalt')
        coeffs = material.extrapolate_coeffs_to_spectrum(interp_degree=1)
        self.assertGreaterEqual(coeffs.all(), 0)
        self.assertLessEqual(coeffs.all(), 1)
        # plt.figure()
        # plt.plot(material.absorption["center_freqs"], material.absorption["coeffs"])
        # plt.show()

        # Test change in fs
        material = Material('average_asphalt')
        fs_test = 96000
        coeffs = material.extrapolate_coeffs_to_spectrum(interp_degree=2, fs = fs_test)
        self.assertAlmostEqual(material.absorption["center_freqs"][-1], fs_test / 2)

    def test_plot(self):
        material = Material('average_asphalt')
        material.extrapolate_coeffs_to_spectrum(interp_degree=1)
        material.plot_absorption_coeffs()

if __name__ == '__main__':
    unittest.main()
