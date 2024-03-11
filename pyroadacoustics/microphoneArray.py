import numpy as np
from typing import List, Union
class MicrophoneArray:
    """
    A class that represents a microphone array. The array is defined by its number of microphones
    and their positions

    Attributes
    ----------
    nmics: int
        number of microphones in the array
    mic_positions: np.ndarray
        2D array having dimensions `[nmics, 3]`, containing the `[x,y,z]` cartesian coordinates
        of each microphone in the array
    mic_orientations: np.ndarray or None
        1D Array containing one angular value, in degrees, per each microphone, expressing is orientation. The
        orientation is referred to the positive x axis (i. e. 0 degrees corresponds to a mic oriented towards the positive x axis)
        Default: None (corresponds to 0 degrees)
    dir_pattern:
        str or list of strings, defining the directivity pattern of each microphone. If a single string is passed, all microphones 
        have the same pattern. Supported patterns: omnidirectional, subcardioid, cardioid, supercardioid, hypercardioid, figure 8. Default: omnidirectional
    signals: np.ndarray
        2D array containing the `nimcs` signals acquired by each microphone in the array
    """

    def __init__(
            self,
            mic_positions: np.ndarray,
            mic_orientations: Union[np.ndarray, None] = None,
            dir_pattern: Union[List[str], str] = 'omnidirectional'
        ) -> None:
        """
        Create microphone array object by defining the positions of the microphones contained the array

        Parameters
        ----------
        mic_positions: np.ndarray
            2D array having dimensions `[nmics, 3]`, containing the `[x,y,z]` cartesian coordinates
            of each microphone in the array
        mic_orientations: np.ndarray
            1D Array containing one angular value, in degrees, per each microphone, expressing is orientation. The
            orientation is referred to the positive x axis (i. e. 0 degrees corresponds to a mic oriented towards the positive x axis)
        dir_pattern:
            str or list of strings, defining the directivity pattern of each microphone. If a single string is passed, all microphones 
            have the same pattern. Supported patterns: omnidirectional, subcardioid, cardioid, supercardioid, hypercardioid, figure 8

        Raises
        ------
        ValueError:
            If mic_positions is empty or contains less than 3 coordinates per microphone
        ValueError:
            If length of mic_orientations is different from length of mic_positions
        ValueError:
            If length of dir_pattern is different from 1 or length of mic_positions

        Modifies
        --------
        nmics
            The attribute is set equal to the first dimension of mic_positions (expressing the number of
            microhones in the array)
        """    
        if mic_positions.any() == None or np.shape(mic_positions)[0] == 0 or True in [len(i)!= 3 for i in mic_positions]:
            raise ValueError("mic_positions must contain at least one microphone. "
            "mic_positions must have shape [n, 3].")
        if mic_orientations is None:
            mic_orientations = np.zeros_like(mic_positions)
        if len(mic_positions) != len(mic_orientations):
            raise ValueError("Length of mic_orientations should match length of mic_positions")
        if len(dir_pattern) != len(mic_positions) and len(dir_pattern) != 1 and not isinstance(dir_pattern, str):
            raise ValueError("dir_pattern must either be a string, or one pattern (i.e. one string) must be passed per each microphone")
        
        self.mic_positions = np.array(mic_positions)
        self.mic_orientations = np.array(mic_orientations)
        if len(dir_pattern) == 1:
            self.dir_pattern = dir_pattern * len(mic_positions)
        elif isinstance(dir_pattern, str):
            self.dir_pattern = [dir_pattern] * len(mic_positions)
        else:
            self.dir_pattern = dir_pattern
        self.nmics = np.shape(mic_positions)[0]
        self.signals = None
