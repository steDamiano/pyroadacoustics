import unittest
import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyroadacoustics.delayLine import DelayLine

class DelayLineTest(unittest.TestCase):
    def test_instantiate(self):
        # Instantiate with 0 samples
        self.assertRaises(ValueError, DelayLine, N = 0)

        # Instantiate with wrong interpolation method
        self.assertRaises(ValueError, DelayLine, interpolation = 'Random')

        # Instantiate with wrong number of read pointers
        self.assertRaises(ValueError, DelayLine, num_read_ptrs = -2)

        # Instantiate and check delay line length
        dl = DelayLine(N = 100, interpolation= 'Sinc', num_read_ptrs = 5)
        self.assertIsInstance(dl, DelayLine)
        self.assertIsInstance(dl.delay_line, np.ndarray)
        self.assertIsInstance(dl.read_ptr, np.ndarray)
        self.assertEqual(len(dl.delay_line), 100)
    
    def test_setdelays(self):
        # Set wrong number of delays
        dl = DelayLine(num_read_ptrs = 2)
        self.assertRaises(ValueError, dl.set_delays, (np.array([10])))

        # Set negative delays
        self.assertRaises(ValueError, dl.set_delays, (np.array([-5, 2])))
        self.assertRaises(ValueError, dl.set_delays, (np.array([5, 0])))

        # Check read pointer assignment
        dl.set_delays(np.array([10, 1]))
        self.assertEqual(dl.read_ptr[0], dl.N - 10)
        self.assertEqual(dl.read_ptr[1], dl.N - 1)

        # Check delay longer than delay line itself
        self.assertRaises(RuntimeError, dl.set_delays, np.array([78000, 2]))

        # Set fractional delay
        dl = DelayLine()
        delay = np.array([1.29481])
        dl.set_delays(delay)
        self.assertAlmostEqual(dl.read_ptr, dl.N - delay)

    def test_update_delayline(self):
        # Read pointer crosses the zero
        dl = DelayLine(interpolation= "Linear")
        dl.set_delays(np.array([1.4]))
        for i in range(10):
            dl.update_delay_line(1, np.array([5]))
            self.assertGreaterEqual(float(dl.read_ptr), 0)
            self.assertLessEqual(float(dl.read_ptr), dl.N)
        
        # Write pointer crosses the zero
        dl = DelayLine(N = 10, interpolation= "Linear")
        for i in range(15):
            dl.update_delay_line(1, np.array([3 - 0.02 * i]))
            # print("Write Pointer: ", dl.write_ptr)
            self.assertGreaterEqual(dl.write_ptr, 0)
            self.assertLessEqual(float(dl.read_ptr), dl.N)

        # Check written values
        dl = DelayLine(N = 10, interpolation= "Linear")
        dl.set_delays(np.array([5]))
        expected_out = np.array([0,0,0,0,0,1,2,3,4,5,6,7,8,9,10])
        for i in range(15):
            out = dl.update_delay_line(i + 1, np.array([5]))
            self.assertEqual(out, expected_out[i])
        
        # Check Linear Interpolation
        dl = DelayLine(N = 10, interpolation= "Linear")
        dl.set_delays(np.array([1]))
        for i in range(15):
            out = dl.update_delay_line(1, np.array([1.3]))
            if i == 1:
                self.assertAlmostEqual(float(out), 0.3 * 0 + 0.7 * 1)
        
        # Check Lagrange Interpolation filter
        dl = DelayLine(interpolation='Lagrange')
        self.assertTrue(np.array_equal(dl._frac_delay_lagrange(1, 0.2), np.array([0.8, 0.2]), equal_nan=True))
        self.assertTrue(np.allclose(dl._frac_delay_lagrange(3, 0.2), np.array([0.672, 0.504, -0.224, 0.048])))

        # Check Allpass Interpolation filter
        dl = DelayLine(N = 10, interpolation='Allpass')
        dl.set_delays(np.array([1]))
        for i in range(15):
            out = dl.update_delay_line(1, np.array([1.3]))
            if i == 1:
                self.assertAlmostEqual(float(out), 0 + 0.7 * (1 - 0))
            if i == 2:
                self.assertAlmostEqual(float(out), 1 + 0.7 * (1 - 0.7))
            if i == 3:
                self.assertAlmostEqual(float(out), 1 + 0.7 * (1 - 1.21))

        # Check Sinc Interpolation filter
        dl = DelayLine(interpolation='Sinc')
        filt = dl._frac_delay_sinc(5, np.hanning(5), 0.2)
        self.assertTrue(np.allclose(filt, np.array([ 0., -0.08, 0.96, 0.12, -0.])))

        # Test computational time
        dl = DelayLine(N = 48000, num_read_ptrs=2)
        dl.set_delays(np.array([250, 370]))

        signal = np.random.randn(40000)
        for i in range(len(signal)):
            out1 = dl.update_delay_line(signal[i], np.array([250 - 0.02 * i, 370 - 0.03 * i]))
    
    def test_sinc_table(self):
        dl = DelayLine(interpolation='Sinc')
        filt_true = dl._frac_delay_sinc(11, np.hanning(11), 0.261)
        filt_int = dl._frac_delay_interpolated_sinc(0.261)
        self.assertTrue(np.linalg.norm(filt_true - filt_int) < 0.04)
        print(filt_true)
        print(filt_int)

if __name__ == '__main__':
    import cProfile
    # cProfile.run('unittest.main()')
    unittest.main()