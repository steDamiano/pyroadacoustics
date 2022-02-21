import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import unittest
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyroadacoustics.environment import Environment
from pyroadacoustics.simulatorManager import SimulatorManager

class SimulatorManagerTest(unittest.TestCase):
    # def test_instantiate(self):
    #     env = Environment(fs = 8000, temperature=20, pressure=1, rel_humidity=50)
    #     env.add_source(position=np.array([0., 0., 1.]))
    #     env.add_microphone_array(np.array([[0,1,1]]))
    #     manager = SimulatorManager(env.c, env.fs, env.Z0, env.road_material, env.air_absorption_coefficients)
    
    # def test_compute_angle(self):
    #     # Equal height case
    #     src_pos = np.array([0., 0., 1.])
    #     mic_pos = np.array([[0,2,1]])
    #     env = Environment(fs = 8000, temperature=20, pressure=1, rel_humidity=50)
    #     env.add_source(position=src_pos)
    #     env.add_microphone_array(mic_pos)
    #     manager = SimulatorManager(env.c, env.fs, env.Z0, env.road_material, env.air_absorption_coefficients)
    #     theta = manager._compute_angle(src_pos, mic_pos[0])
    #     self.assertAlmostEqual(theta, np.pi / 4)

    #     # Limit case: one source on the ground
    #     src_pos = np.array([0., 0., 1.])
    #     mic_pos = np.array([[0,1,0.000001]])
    #     theta = manager._compute_angle(src_pos, mic_pos[0])
    #     self.assertAlmostEqual(theta, np.pi/4,4)

    #     # Arbitrary case (hand computation to check)
    #     src_pos = np.array([0., 0., 1.])
    #     mic_pos = np.array([[0,2,4]])
    #     theta = manager._compute_angle(src_pos, mic_pos[0])
    #     self.assertAlmostEqual(theta, 1.19028994968)

    # def test_compute_delay(self):
    #     src_pos = np.array([0., 0., 1.])
    #     mic_pos = np.array([[0,2,1]])
    #     env = Environment(fs = 8000, temperature=20, pressure=1, rel_humidity=50)
    #     env.add_source(position=src_pos)
    #     env.add_microphone_array(mic_pos)
    #     manager = SimulatorManager(env.c, env.fs, env.Z0, env.road_material, env.air_absorption_coefficients)
    #     d, tau = manager._compute_delay(src_pos, mic_pos)
    #     self.assertEqual(d, 2)
    #     self.assertAlmostEqual(tau, d/env.c)

    #     # Coincident positions
    #     src_pos = np.array([0., 0., 1.])
    #     mic_pos = np.array([[0,0,1]])
    #     d, tau = manager._compute_delay(src_pos, mic_pos)
    #     self.assertEqual(d, 0)
    #     self.assertAlmostEqual(tau, 0)
    
    # def test_angle_reflection_table(self):
    #     src_pos = np.array([0., 0., 1.])
    #     mic_pos = np.array([[0,2,1]])
    #     env = Environment(fs = 8000, temperature=20, pressure=1, rel_humidity=50)
    #     env.add_source(position=src_pos)
    #     env.add_microphone_array(mic_pos)
    #     manager = SimulatorManager(env.c, env.fs, env.Z0, env.road_material, env.air_absorption_coefficients)
    #     table = manager.asphaltReflectionFilterTable
    #     self.assertTrue(np.allclose(np.array([6.68434673e-05, 6.89725198e-05, 8.77270776e-04, 1.10158154e-03,
    #         1.79727107e-02, 9.05203068e-01, 1.79727107e-02, 1.10158154e-03,
    #         8.77270776e-04, 6.89725198e-05, 6.68434673e-05]), table[89], rtol=0.0001))
       
    #     # w, H_fir = scipy.signal.freqz(table[89], a=1, worN = 11)
    #     # plt.plot(w, abs(H_fir))
    #     # plt.show()
    
    # def test_get_reflection_filter(self):
    #     src_pos = np.array([0., 0., 1.])
    #     mic_pos = np.array([[0,2,1]])
    #     env = Environment(fs = 8000, temperature=20, pressure=1, rel_humidity=50)
    #     env.add_source(position=src_pos)
    #     env.add_microphone_array(mic_pos)
    #     manager = SimulatorManager(env.c, env.fs, env.Z0, env.road_material, env.air_absorption_coefficients)
    #     table = manager.asphaltReflectionFilterTable
    #     self.assertTrue(np.allclose(manager._get_asphalt_reflection_filter(0), table[89]))
    
    # def test_air_absorption_filter(self):
    #     src_pos = np.array([0., 0., 1.])
    #     mic_pos = np.array([[0,2,1]])
    #     env = Environment(fs = 8000, temperature=20, pressure=1, rel_humidity=50)
    #     env.add_source(position=src_pos)
    #     env.add_microphone_array(mic_pos)
    #     manager = SimulatorManager(env.c, env.fs, env.Z0, env.road_material, env.air_absorption_coefficients)
    #     airabs = manager._compute_air_absorption_filter(2, 11)
    #     self.assertTrue(np.allclose(airabs, np.array([ 1.77146741e-05, -4.40299508e-05, 2.17979459e-04, -7.52579305e-04,
    #         4.40111427e-03,  9.91346868e-01,  4.40111427e-03, -7.52579305e-04, 2.17979459e-04,
    #         -4.40299508e-05,  1.77146741e-05])))
    
    # def test_set_parameters(self):
    #     env = Environment(fs = 8000, temperature=20, pressure=1, rel_humidity=50)

    #     params = {
    #         "interp_method": "Linear",
    #         "prova": 0
    #     }

    #     self.assertRaises(KeyError, SimulatorManager, env.c, env.fs, env.Z0, env.road_material, 
    #         env.air_absorption_coefficients, params)

    #     params = {
    #             "interp_method": "Linear",
    #             "include_reflected_path": True,
    #             "include_air_absorption": False,
    #     }

    #     manager = SimulatorManager(env.c, env.fs, env.Z0, env.road_material, env.air_absorption_coefficients, params)
    #     self.assertEqual(manager.simulation_params["interp_method"], "Linear")
        
    # def test_initialize(self):
    #     src_pos = np.array([0., 0., 1.])
    #     mic_pos = np.array([[0,2,1]])
    #     env = Environment(fs = 8000, temperature=20, pressure=1, rel_humidity=50)
    #     env.add_source(position=src_pos)
    #     env.add_microphone_array(mic_pos)
    #     manager = SimulatorManager(env.c, env.fs, env.Z0, env.road_material, env.air_absorption_coefficients)
    #     manager.initialize(src_pos, mic_pos[0])

    #     self.assertAlmostEqual(manager.primaryDelLine.read_ptr[0], 48000 - 2 / env.c * env.fs)
    #     self.assertAlmostEqual(manager.primaryDelLine.read_ptr[1], 48000 - np.sqrt(2) / env.c * env.fs)
    #     self.assertAlmostEqual(manager.secondaryDelLine.read_ptr[0], 48000 - np.sqrt(2) / env.c * env.fs)
    
    def test_update(self):
        # Initialization
        src_pos = np.array([0., 2, 1.])
        mic_pos = np.array([[0,0,1]])
        
        env = Environment(fs = 8000, temperature=20, pressure=1, rel_humidity=50)
        params = {
            "interp_method": "Linear",
            "include_reflected_path": True,
            "include_air_absorption": True,
        }
        env.set_simulation_params("Sinc", True, True)

        manager = SimulatorManager(env.c, env.fs, env.Z0, env.road_material, env.air_absorption_coefficients, 
            env.simulation_params)

        env.add_source(position=src_pos)
        env.add_microphone_array(mic_pos)
        manager.initialize(src_pos, mic_pos[0])

        y_received = np.zeros(1000)
        for i in range(1000):
            y_received[i] = manager.update(src_pos, mic_pos[0], 1)
            # print(y_received[i])

        self.assertTrue(np.allclose(manager.primaryDelLine.delay_line[0:1000], np.ones(1000)))
        self.assertTrue(max(y_received) < 1)
        
if __name__ == '__main__':
    import cProfile
    cProfile.run('unittest.main()')
