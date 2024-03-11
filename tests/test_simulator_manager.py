import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import unittest
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyroadacoustics.material import Material
from pyroadacoustics.environment import Environment
from pyroadacoustics.simulatorManager import SimulatorManager

class SimulatorManagerTest(unittest.TestCase):
    def test_instantiate(self):
        env = Environment(fs = 8000, temperature=20, pressure=1, rel_humidity=50)
        env.add_source(position=np.array([0., 0., 1.]))
        env.add_microphone_array(np.array([[0,1,1]]))
        manager = SimulatorManager(env.c, env.fs, env.Z0, env.road_material, env.air_absorption_coefficients)
    
    def test_compute_angle(self):
        # Equal height case
        src_pos = np.array([0., 0., 1.])
        mic_pos = np.array([[0,2,1]])
        env = Environment(fs = 8000, temperature=20, pressure=1, rel_humidity=50)
        env.add_source(position=src_pos)
        env.add_microphone_array(mic_pos)
        manager = SimulatorManager(env.c, env.fs, env.Z0, env.road_material, env.air_absorption_coefficients)
        theta = manager._compute_angle(src_pos, mic_pos[0])
        self.assertAlmostEqual(theta, np.pi / 4)

        # Limit case: one source on the ground
        src_pos = np.array([0., 0., 1.])
        mic_pos = np.array([[0,1,0.000001]])
        theta = manager._compute_angle(src_pos, mic_pos[0])
        self.assertAlmostEqual(theta, np.pi/4,4)

        # Arbitrary case (hand computation to check)
        src_pos = np.array([0., 0., 1.])
        mic_pos = np.array([[0,2,4]])
        theta = manager._compute_angle(src_pos, mic_pos[0])
        self.assertAlmostEqual(theta, 1.19028994968)

    def test_compute_delay(self):
        src_pos = np.array([0., 0., 1.])
        mic_pos = np.array([[0,2,1]])
        env = Environment(fs = 8000, temperature=20, pressure=1, rel_humidity=50)
        env.add_source(position=src_pos)
        env.add_microphone_array(mic_pos)
        manager = SimulatorManager(env.c, env.fs, env.Z0, env.road_material, env.air_absorption_coefficients)
        d, tau = manager._compute_delay(src_pos, mic_pos)
        self.assertEqual(d, 2)
        self.assertAlmostEqual(tau, d/env.c)

        # Coincident positions
        src_pos = np.array([0., 0., 1.])
        mic_pos = np.array([[0,0,1]])
        d, tau = manager._compute_delay(src_pos, mic_pos)
        self.assertEqual(d, 0)
        self.assertAlmostEqual(tau, 0)
    
    def test_angle_reflection_table(self):
        src_pos = np.array([0., 0., 1.])
        mic_pos = np.array([[0,2,1]])
        env = Environment(fs = 8000, temperature=20, pressure=1, rel_humidity=50, road_material=Material('average_asphalt'))
        env.add_source(position=src_pos)
        env.add_microphone_array(mic_pos)
        manager = SimulatorManager(env.c, env.fs, env.Z0, env.road_material, env.air_absorption_coefficients)
        table = manager._precompute_real_angle_reflection_filter_table(11)
        self.assertTrue(np.allclose(np.array([2.04541068e-04, 6.54709799e-04, 1.95861242e-03, 5.61281019e-03,
       4.46539850e-03, 9.33648475e-01, 4.46539850e-03, 5.61281019e-03,
       1.95861242e-03, 6.54709799e-04, 2.04541068e-04]), table[89]))
       
    
    def test_get_reflection_filter(self):
        src_pos = np.array([0., 0., 1.])
        mic_pos = np.array([[0,2,1]])
        env = Environment(fs = 8000, temperature=20, pressure=1, rel_humidity=50, road_material=Material('average_asphalt'))
        env.add_source(position=src_pos)
        env.add_microphone_array(mic_pos)
        manager = SimulatorManager(env.c, env.fs, env.Z0, env.road_material, env.air_absorption_coefficients)
        table = manager.realAsphaltReflectionFilterTable
        self.assertTrue(np.allclose(manager._get_asphalt_reflection_filter(0,10), table[89]))
    
    def test_air_absorption_filter(self):
        src_pos = np.array([0., 0., 1.])
        mic_pos = np.array([[0,2,1]])
        env = Environment(fs = 8000, temperature=20, pressure=1, rel_humidity=50)
        env.add_source(position=src_pos)
        env.add_microphone_array(mic_pos)
        manager = SimulatorManager(env.c, env.fs, env.Z0, env.road_material, env.air_absorption_coefficients)
        airabs = manager._compute_air_absorption_filter(2, 11)
        
        # FIRLS Method
        alpha = manager.airAbsorptionCoefficients
        alpha_lin = 10 ** (-2 * alpha / 20)
        airabs2 = scipy.signal.firwin2(11, np.linspace(0,1,20), alpha_lin)
        self.assertTrue(np.linalg.norm(airabs - airabs2) < 0.01)

    
    def test_set_parameters(self):
        env = Environment(fs = 8000, temperature=20, pressure=1, rel_humidity=50)

        params = {
            "interp_method": "Linear",
            "prova": 0
        }

        self.assertRaises(KeyError, SimulatorManager, env.c, env.fs, env.Z0, env.road_material, 
            env.air_absorption_coefficients, params)

        params = {
                "interp_method": "Linear",
                "include_reflected_path": True,
                "include_air_absorption": False,
        }

        manager = SimulatorManager(env.c, env.fs, env.Z0, env.road_material, env.air_absorption_coefficients, params)
        self.assertEqual(manager.simulation_params["interp_method"], "Linear")
        
    def test_initialize(self):
        src_pos = np.array([0., 0., 1.])
        mic_pos = np.array([[0,2,1]])
        env = Environment(fs = 8000, temperature=20, pressure=1, rel_humidity=50)
        env.add_source(position=src_pos)
        env.add_microphone_array(mic_pos)
        manager = SimulatorManager(env.c, env.fs, env.Z0, env.road_material, env.air_absorption_coefficients)
        manager.initialize(env.source.trajectory, mic_pos[0], env.mic_array.mic_orientations[0,0],env.mic_array.dir_pattern[0],env.source.src_orientation,env.source.dir_pattern)

        self.assertAlmostEqual(manager.primaryDelLine.read_ptr[0], 48000 - 2 / env.c * env.fs)
        self.assertAlmostEqual(manager.primaryDelLine.read_ptr[1], 48000 - np.sqrt(2) / env.c * env.fs)
        self.assertAlmostEqual(manager.secondaryDelLine.read_ptr[0], 48000 - np.sqrt(2) / env.c * env.fs)
    
    # def test_retrieve_absorption_coeffs(self):
    #     # Initialization
    #     src_pos = np.array([0., 2, 1.])
    #     mic_pos = np.array([[0,0,1]])
        
    #     env = Environment(fs = 8000, temperature=20, pressure=1, rel_humidity=50)
    #     params = {
    #         "interp_method": "Allpass",
    #         "include_reflected_path": True,
    #         "include_air_absorption": True,
    #     }
    #     env.set_simulation_params("Allpass", True, True)

    #     manager = SimulatorManager(env.c, env.fs, env.Z0, env.road_material, env.air_absorption_coefficients, 
    #         env.simulation_params)

    #     env.add_source(position=src_pos)
    #     env.add_microphone_array(mic_pos)
    #     manager.initialize(src_pos, mic_pos[0])

    #     new_points = np.arange(0,100,1)
    #     new_coeff1 = np.polyval(manager._model[5], new_points)

    #     coeffs_true = manager._compute_air_absorption_filter(2, 11)
    #     coeffs_approx = manager._retrieve_air_absorption_filter(2)
    #     self.assertTrue(np.allclose(coeffs_true, coeffs_approx, rtol = 0.001))
        
    #     # plt.plot(new_points, new_coeff1)
    #     # plt.plot(2, coeffs_true[5], 'gx')
    #     # plt.plot(2, coeffs_approx[5], 'ro')
    #     # plt.show()
        
    def test_update(self):
        
        # Initialization
        src_pos = np.array([0., 2, 1.])
        mic_pos = np.array([[0,0,1]])
        
        env = Environment(fs = 8000, temperature=20, pressure=1, rel_humidity=50)
        params = {
            "interp_method": "Allpass",
            "include_reflected_path": True,
            "include_air_absorption": True,
        }
        env.set_simulation_params("Allpass", True, True)

        manager = SimulatorManager(env.c, env.fs, env.Z0, env.road_material, env.air_absorption_coefficients, 
            env.simulation_params)

        env.add_source(position=src_pos)
        env.add_microphone_array(mic_pos)
        manager.initialize(env.source.trajectory, mic_pos[0], env.mic_array.mic_orientations[0,0],env.mic_array.dir_pattern[0],env.source.src_orientation,env.source.dir_pattern)
        import time
        start_time = time.time()
        import cProfile
        pr = cProfile.Profile()
        pr.enable()
        y_received = np.zeros(1000)
        
        for i in range(1000):
            y_received[i] = manager.update(src_pos, mic_pos[0], 1)
        pr.disable()
        pr.print_stats(sort = 'time')
        self.assertTrue(np.allclose(manager.primaryDelLine.delay_line[0:1000], np.ones(1000)))
        self.assertTrue(max(y_received) < 1)
        
if __name__ == '__main__':
    import cProfile
    # cProfile.run('unittest.main()')
    unittest.main()
    # tst = SimulatorManagerTest()
    # tst.test_initialize()
