import numpy as np

class SimulatorManager:

    __instance = None
    @staticmethod
    def getInstance():
        if SimulatorManager.__instance == None:
            SimulatorManager()
        return SimulatorManager.__instance

    def __init__(self, source = None, mic_array = None, fs = 8000) -> None:
        if SimulatorManager.__instance != None:
            raise Exception("SimulatorManager already instantiated")
        else:
            SimulatorManager.__instance = self

        self.src = source
        self.mic = mic_array
        self.fs = fs
        self.c = 343                    # Speed of sound in air
        self.T = 25                     # Temperature in Celsius
        self.p = 1                      # Pressure in atmospheres
        self.active_microphone = 0      # Start simulations from Microphone zero

    def run_simulation(self):
        for i in range(len(self.mic)):
            self.active_microphone = i

            self.initialize()
            self.update()

    def compute_initial_dstance(self):
        # Assume source and receiver lie on the same plane above the ground
        height = 1

        # Compute distance from MIC (origin) to line
        dline = np.abs((self.source.trajectory[1][0] - self.source.trajectory[0][0]) * 
            (self.source.trajectory[0][1] - self.mic[self.active_microphone][1]) - 
            (self.source.trajectory[0][0] - self.mic[self.active_microphone][0]) * 
            (self.source.trajectory[1][1] - self.source.trajectory[0][1])) / np.sqrt(np.sum((self.source.trajectory[1] - 
            self.source.trajectory[0])**2))
        # dline2 = dline ** 2

        # Compute distance from P0 to projection of MIC on the line
        l2 = np.sum((self.source.trajectory[0]-self.source.trajectory[1])**2)
        t = np.sum((self.mic[self.active_microphone] - self.source.trajectory[0]) * 
            (self.source.trajectory[1] - self.source.trajectory[0])) / l2
        P_proj = self.source.trajectory[0] + t * (self.source.trajectory[1] - self.source.trajectory[0])
        dproj_init = np.sqrt(np.sum((self.source.trajectory[0] - P_proj) ** 2))

        # Compute initial distance from MIC to P1
        dinit = np.sqrt(dproj_init ** 2 + dline ** 2)

        # Compute initial distance from first reflection to MIC
        dinit_refl = np.sqrt(dinit ** 2 + (2 * height) ** 2)

        # Compute time delays in number of samples
        tau_init = dinit / self.c
        tau_init_refl = dinit_refl / self.c

        M_init = tau_init * self.fs
        M_init_refl = tau_init_refl * self.fs

        print('Initial delay in samples - direct: %f' %M_init)
        print('Initial delay in samples - reflected: %f' %(M_init_refl / 2))

    def initialize(self):
        self.compute_initial_dstance()
    

    def update():
        pass
