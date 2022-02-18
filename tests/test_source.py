import matplotlib.pyplot as plt
import numpy as np
import unittest

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyroadacoustics.soundSource import SoundSource

class SoundSourceTest(unittest.TestCase):
    def test_instantiate(self):
        # Static Source
        src = SoundSource(np.array([1,3,1]), fs = 8000, is_static=True, static_simduration = 5)
        self.assertEqual(np.shape(src.trajectory), (8000 * 5, 3))

        src = SoundSource(np.array([1,3,1]), fs = 8000, is_static=True, static_simduration = 0.001)
        self.assertEqual(np.shape(src.trajectory), (round(8000 * 0.001), 3))

    def test_define_trajectory(self):
        # Static Source
        src = SoundSource(is_static= True)
        self.assertRaises(RuntimeError, src.set_trajectory, np.array([[1,1,1],[1,2,1]]), np.array([1]))

        # Moving Source
        
        # Wrong number of speed values
        src = SoundSource(is_static= False)
        self.assertRaises(ValueError, src.set_trajectory, np.array([[1,1,1],[1,2,1]]), np.array([1,3]))

        # Constant Speed
        src = SoundSource(is_static= False)
        src.set_trajectory(np.array([[1,1,1],[1,2,1], [5,2,1]]),np.array([1]))
        self.assertEqual(np.shape(src.trajectory)[1], 3)
        self.assertIn(np.array([1,1,1]), src.trajectory)
        self.assertIn(np.array([1,2,1]), src.trajectory)
        self.assertIn(np.array([5,2,1]), src.trajectory)
        
        # plt.figure()
        # plt.plot(src.trajectory[:,0], src.trajectory[:,1])
        # plt.show()

        # Zero speed
        src = SoundSource(is_static= False)
        self.assertRaises(ValueError, src.set_trajectory, np.array([[1,1,1],[1,2,1], [5,2,1]]),np.array([1, 0]))
        self.assertRaises(ValueError, src.set_trajectory, np.array([[1,1,1],[1,2,1], [5,2,1]]),np.array([0]))

        # Coincident trajectory points
        src = SoundSource(is_static= False)
        src.set_trajectory(np.array([[1,1,1],[1,1,1], [5,2,1]]),np.array([1]))
        # plt.figure()
        # plt.plot(src.trajectory[:,0], src.trajectory[:,1])
        # plt.show()
    
    def test_set_signal(self):
        dummy_signal = np.sin(2 * np.pi * 400 * np.arange(0,2, 1/8000))
        src = SoundSource(np.array([0,3,5]), fs = 8000, is_static=True, static_simduration=5)
        src.set_signal(dummy_signal)
        self.assertEqual(np.shape(dummy_signal), np.shape(src.signal))
        


if __name__ == '__main__':
    unittest.main()