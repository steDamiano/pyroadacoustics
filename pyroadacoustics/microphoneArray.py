import numpy as np
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
    signals: np.ndarray
        2D array containing the `nimcs` signals acquired by each microphone in the array
    """

    def __init__(
            self,
            mic_positions,
        ) -> None:
        """
        Create microphone array object by defining the positions of the microphones contained the array

        Parameters
        ----------
        mic_positions: np.ndarray
            2D array having dimensions `[nmics, 3]`, containing the `[x,y,z]` cartesian coordinates
            of each microphone in the array

        Raises
        ------
        ValueError:
            If mic_positions is empty or contains less than 3 coordinates per microphone

        Modifies
        --------
        nmics
            The attribute is set equal to the first dimension of mic_positions (expressing the number of
            microhones in the array)
        """    
        if mic_positions.any() == None or np.shape(mic_positions)[0] == 0 or True in [len(i)!= 3 for i in mic_positions]:
            raise ValueError("mic_positions must contain at least one microphone. "
            "mic_positions must have shape [n, 3].")
        self.mic_positions = np.array(mic_positions)
        self.nmics = np.shape(mic_positions)[0]
        self.signals = None
