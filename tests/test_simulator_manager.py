import numpy as np
import unittest
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Environment import Environment
from src.SimulatorManager import SimulatorManager

class SimulatorManagerTest(unittest.TestCase):
    def test_instantiate(self):
        pass

if __name__ == '__main__':
    unittest.main()