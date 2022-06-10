import numpy as np

class SoundSource:
    """
    A class that defines a sound source. The sound source is assumed to be point-like and 
    omnidirectional, and emits an arbitrary sound signal. It is initially located in a 
    defined position, and moves along an arbitrary trajectory defined as a series of segments
    connecting a set of points, given as input.

    Attributes
    ----------
    position: np.ndarray
        1D array that contains initial position of sound source as a set of 3 cartesian coordinates `[x,y,z]`
    signal: np.ndarray
        1D array that contains samples of signal emitted by the sound source
    fs: int
        Sampling frequency of the emitted signal
    is_static: bool
        True if the source is static during the simulation
    static_simduration: int
        If the source is static, defines the duration of the simulations in seconds. If the source moves,
        the simulation duration is defined by the time it takes to travel the whole trajectory

    Methods
    -------
    set_trajectory(positions, speed):
        Defines a trajectory from a given set of N positions (`positions`) and N-1 velocities (`speed`). The speed
        is assumed to be constant between each couple of positions.
    set_signal(signal):
        Setter for signal attribute: assigns given signal to the sound source
    """

    def __init__(
            self,
            position = np.array([0,0,1]),
            fs = 8000,
            is_static = True,
            static_simduration = 5
        ) -> None:
        """
        Create SoundSource object by defining its initial position, trajectory, 
        emitted signal and sampling frequency.

        Parameters
        ----------
        position : ndarray
            1D Array containing 3 cartesian coordinates `[x,y,z]` that define initial source position, 
            by default [0,0,1]
        fs : int, optional
            Sampling frequency of the emitted signal, by default 8000
        is_static: Bool, optional
            True if the source is static
        static_simduration: float, optional
            If the source is static, defines the duration of the simulations in seconds. If the source moves,
            the simulation duration is defined by the time it takes to travel the whole trajectory
        """

        self.position = position
        self.signal = None
        self.fs = fs
        self.is_static = is_static
        self.static_simduration = static_simduration
        if self.is_static:
            # If duration is zero, consider single sample
            if self.static_simduration == 0:
                self.static_simduration = 1 / self.fs
            self.trajectory = np.tile(self.position, (round(self.fs * self.static_simduration), 1))
        
    def set_signal(self, signal: np.ndarray) -> None:
        """
        Setter for signal attribute: assigns given signal to the sound source

        Parameters
        ----------
        signal : np.ndarray
            1D Array containing samples of the source signal
        """
        self.signal = signal

    def set_trajectory(self, positions: np.ndarray, speed: np.ndarray) -> np.ndarray:
        """
        Defines a trajectory for the sound source from a set of N positions (given as triplets of Cartesian
        coordinates) and the values of the modulus of the source velocity between each subsequent couple of positions.
        The trajectory is defined as the positions assumed by the sound source at each sample of the simulation.
        Therefore, given the N input points, the method first computes a set of N-1 segments connecting one point to 
        the following one. Then, each segment is sampled according to the speed of the source in that segment and
        the signal sampling frequency, to yield a series of points so that each signal sample is emitted by the
        source at a different position of the trajectory.

        Parameters
        ----------
        positions : np.ndarray
            2D Array containing N sets of 3 cartesian coordinates `[x,y,z]` defining the desired trajectory positions.
            Each couple of subsequent points defines a straight segment on the overall trajectory
        speed : np.ndarray
            * 2D Array containing N-1 floats defining the modulus of the velocity on each trajectory segment
            * 1D Array containing one float, defining the modulus of the velocity on the whole trajectory (i.e. constant speed)

        Returns
        -------
        trajectory: np.ndarray
            2D Array containing N sets of 3 cartesian coordinates `[x,y,z]` defining the full sampled trajectory

        Raises
        ------
        ValueError
            If `speed` is neither a `float` nor a `np.ndarray` so that `len(speed) != (np.shape(positions)[0] - 1)`
        ValueError
            If `speed` is 0 or speed array contains value 0

        Modifies
        --------
        trajectory
            Parameter is updated with the new computed trajectory

        """

        if self.is_static:
            raise RuntimeError("Cannot assign trajectory to static source")

        trajectory = np.empty((0,3), dtype = np.float64)
        if len(speed) != (np.shape(positions)[0] - 1):
            if(len(speed) != 1):
                raise ValueError('Speed must be an array with len(speed) = np.shape(positions)[0] - 1 or len(speed) == 1!')
            # else:
                # Tile the speed value to cover the whole set of segments
                # speed = speed * np.ones(np.shape(positions)[0], 1)
                # speed = np.tile(speed, np.shape(positions)[0] - 1)
        if 0 in speed:
            raise ValueError("Speed cannot be zero")
        if len(speed) == 1:
            cum_lengths = np.zeros(len(positions) + 1)

            len_traj = 0
            b_seg = positions[0]
            

            for i in range(0, len(positions)):
                e_seg = positions[i]
                len_traj += np.sqrt((e_seg[0] - b_seg[0])**2 + (e_seg[1] - b_seg[1])**2)
                cum_lengths[i] = len_traj
                b_seg = e_seg
            
            cum_lengths[-1] = 2 * len_traj
            sim_time = len_traj / speed[0]

            num_points = int(sim_time * self.fs - 1)
            # trajectory = np.zeros((num_points,3))

            curr = 0
            next = 1

            for i in range(0, num_points):
                z = len_traj * i / num_points

                while(z > cum_lengths[next]):
                    curr += 1
                    next += 1
                if np.allclose(positions[curr], positions[next]):
                    curr += 1
                    next += 1
                
                b_seg = positions[curr]
                e_seg = positions[next]

                t = (z - cum_lengths[curr]) / (cum_lengths[next] - cum_lengths[curr])

                trajectory = np.append(trajectory, np.array([[(b_seg[0] * (1 - t) + e_seg[0] * t), b_seg[1] * (1 - t) + e_seg[1] * t, 1]]), axis = 0)

            # res[-1] = (seg_line[-1])
        else:
            for i in range(1, np.shape(positions)[0]):
                # Extremes of the considered segment
                a = positions[i - 1]
                b = positions[i]

                # Direction defining segment passing for a and b
                direction = b - a
                direction = direction / np.linalg.norm(direction)

                len_segment = np.sqrt(np.sum((a-b)**2))         # Compute length of segment A,B
                t_segment = len_segment / speed[i-1]            # Time to go from A to B (seconds)
                samples_segment = round(t_segment * self.fs)    # Number of samples to go from A to B

                # Positions on segment at each sample
                segment_positions = len_segment / samples_segment * range(samples_segment)
                # segment_positions = np.append(segment_positions, 1).reshape(-1,1)
                segment_positions = segment_positions.reshape(-1,1)
            
                segment_positions = np.tile(a,(len(segment_positions), 1)) + segment_positions * direction
                trajectory = np.append(trajectory, segment_positions, axis = 0)
            
        trajectory = np.append(trajectory, np.reshape(positions[-1], (1,3)), axis = 0)
        self.trajectory = trajectory

        return trajectory