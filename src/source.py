import numpy as np
class Source:
    def __init__(self, position = None, velocity = None, signal = None, trajectory = None):
        self.position = position
        self.trajectory = trajectory
        self.velocity = velocity
        self.signal = signal
    
    def set_trajectory(self, P1: np.array, P2: np.array):
        # Define rectilinear trajectory on the line defined by P1 and P2. 
        # Starting Position is P1.
        self.position = P1
        self.trajectory = np.concatenate((P1.reshape(1,2), P2.reshape(1,2)))
        
        # # Assume source and receiver lie on the same plane above the ground
        # height = 1
        # v = 5   # speed of sound source

        # # Compute distance from MIC (origin) to line
        # dline = np.abs((P2[0] - P1[0]) * (P1[1] - P3[1]) - (P1[0] - P3[0]) * (P2[1] - P1[1])) / np.sqrt(np.sum((P2-P1)**2))
        # dline2 = dline ** 2

        # # Compute distance from P0 to projection of MIC on the line
        # l2 = np.sum((P1-P2)**2)
        # t = np.sum((P3 - P1) * (P2 - P1)) / l2
        # P_proj = P1 + t * (P2 - P1)
        # dproj_init = np.sqrt(np.sum((P1 - P_proj) ** 2))

        # # Compute initial distance from MIC to P1
        # dinit = np.sqrt(dproj_init ** 2 + dline ** 2)

        # # Compute initial distance from first reflection to MIC
        # dinit_refl = np.sqrt(dinit ** 2 + (2 * height) ** 2)

        # # Compute time delays in number of samples
        # c = 343
        # tau_init = dinit / c
        # tau_init_refl = dinit_refl / c

        # M_init = tau_init * fs
        # M_init_refl = tau_init_refl * fs

        # print('Initial delay in samples - direct: %f' %M_init)
        # print('Initial delay in samples - reflected: %f' %(M_init_refl / 2))
    
    # def set_trajectory(points: np.array) -> None:
    #     raise NotImplementedError("Only rectilinear tracjectories are supported for now")

## TEST CODE
a = Source()
P1 = np.array([3,10])
P2 = np.array([3, -5])
a.set_trajectory(P1, P2)
