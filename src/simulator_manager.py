import numpy as np
import delay_line as dl
import math

class SimulatorManager:

    __instance = None
    A = None
    B = None

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
        self.height = 1                 # Height of Source and Receiver
        self.dline2 = 0                 # Squared distance between microphone and trajectory
        self.dproj = 0

    def run_simulation(self):
        for i in range(np.shape(self.mic.R)[0]):
            self.active_microphone = i

            A, B = self.initialize()
            for j in range(len(self.src.signal)):
                self.update(A, B)

    def compute_initial_dstance(self):

        # Compute distance from MIC (origin) to line
        dline = np.abs((self.src.trajectory[1][0] - self.src.trajectory[0][0]) * 
            (self.src.trajectory[0][1] - self.mic.R[self.active_microphone][1]) - 
            (self.src.trajectory[0][0] - self.mic.R[self.active_microphone][0]) * 
            (self.src.trajectory[1][1] - self.src.trajectory[0][1])) / np.sqrt(np.sum((self.src.trajectory[1] - 
            self.src.trajectory[0])**2))
        dline2 = dline ** 2

        # Compute distance from P0 to projection of MIC on the line
        l2 = np.sum((self.src.trajectory[0]-self.src.trajectory[1])**2)
        t = np.sum((self.mic.R[self.active_microphone] - self.src.trajectory[0]) * 
            (self.src.trajectory[1] - self.src.trajectory[0])) / l2
        P_proj = self.src.trajectory[0] + t * (self.src.trajectory[1] - self.src.trajectory[0])
        dproj_init = np.sqrt(np.sum((self.src.trajectory[0] - P_proj) ** 2))

        # Compute initial distance from MIC to P1
        dinit = np.sqrt(dproj_init ** 2 + dline ** 2)

        # Compute initial distance from first reflection to MIC
        dinit_refl = np.sqrt(dinit ** 2 + (2 * self.height) ** 2)

        # Compute time delays in number of samples
        tau_init = dinit / self.c
        tau_init_refl = dinit_refl / self.c

        M_init = tau_init * self.fs
        M_init_refl = tau_init_refl * self.fs

        print('Initial delay in samples - direct: %f' %M_init)
        print('Initial delay in samples - reflected: %f' %(M_init_refl / 2))
        return M_init, (M_init_refl / 2), dline2, dproj_init

    def initialize(self):
        Mdir, Mrefl, dline2, dproj_init = self.compute_initial_dstance()
        self.dline2 = dline2
        self.dproj = dproj_init
        A = dl.DelayLine(N = 48000, write_ptr = 0, read_ptr = np.array([0,0]), fs = self.fs)
        B = dl.DelayLine(N = 48000, write_ptr = 0, read_ptr = np.array([0]), fs = self.fs)

        A.set_delays(np.array([Mdir, Mrefl]))
        B.set_delays(np.array([Mrefl]))

        return A, B
        
        
    def compute_new_delays(self):
        self.dproj -= self.src.velocity / self.fs
        # dproj_refl = dproj / 2
        d2 = self.dline2 + self.dproj ** 2
        d = np.sqrt(d2)
        drefl2 = d2 + (2 * self.height) ** 2
        drefl = np.sqrt(drefl2)
        # angle of incidence
        cos_theta = d / drefl
        theta = math.degrees(math.acos(cos_theta))
        drefl = drefl / 2

        tau = d / self.c
        tau_refl = drefl / self.c
        return np.array([tau, tau_refl]), theta

    def update(self, A, B):
        tau, theta = self.compute_new_delays()
        A.update_delay_line(1, tau)
        B.update_delay_line(1, np.array([tau[1]]))
        print('Delays A: ')
        print(tau * self.fs)
