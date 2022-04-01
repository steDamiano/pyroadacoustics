import numpy as np
import matplotlib.pyplot as plt
import os, json, io


# Load materials data from json database and store in dict
_materials_database_fn = os.path.join(os.path.dirname(__file__), "materials.json")

with io.open(_materials_database_fn) as database_materials:
    materials_dict = json.load(database_materials)


class Material:
    """
    A class that describes the absorption properties of surfaces.

    Attributes
    ----------
    absorption: dict
        A dictionary containing keys ``description``, ``coeffs`` and ``center_freqs``
    reflection_coeffs: np.ndarray
        An array containing the reflection coefficients of the material in each frequency 
        band defined in the `absorption["center_freqs"]` attribute. Reflection coefficients
        are computed as:

        `refl_coeff[i] = np.sqrt(1 - abs_coeff[i])`
    
    Methods
    -------
    extrapolate_coeffs_to_spectrum(interp_degree = 2, fs = 8000, n_bands = 8):
        Extrapolate absorption coefficients to fill the frequency spectrum from 0 to fs/2
    plot_absorption_coeffs():
        Plot the absorption coefficients in the frequency bands considered in the analysis
    """
    
    def __init__(
            self,
            absorption,
        ) -> None:
        """
        Create the Material object by defining its absorption and reflection properties.

        Parameters
        ----------
        absorption: str or dict
        * str: the absorption coefficients will be obtained from the database
        * dict: a dictionary containing keys ``description``, ``coeffs`` and ``center_freqs``

        Raises
        ------
        KeyError
            If absorption is `dict` but does contain `coeffs` and `center_freqs` keys
        ValueError
            If number of elements in `coeffs` and `center_freqs` is different
        TypeError
            If absorption is neither `dict` nor `str`

        Modifies
        --------
        absorption
            Attribute gets a value depending on input parameter
        reflection_coeffs
            Compute reflection coefficients based on absorption coefficients value
        """

        if isinstance(absorption, str):
            absorption = dict(materials_dict[absorption])
        
        elif isinstance(absorption, dict):
            if "coeffs" not in absorption or "center_freqs" not in absorption:
                raise KeyError("'absorption' must be a dictionary with keys 'coeffs' and 'center_freqs'")

            if len(absorption["coeffs"]) != len(absorption["center_freqs"]):
                raise ValueError("Length of 'absorption['coeffs']' and "
                    "'absorption['center_freqs']' must match.")
        else:
            raise TypeError('Wrong Material Assignment')
        
        self.absorption = absorption
        self.reflection_coeffs = np.sqrt(1 - np.array(self.absorption["coeffs"]))
    
    def extrapolate_coeffs_to_spectrum(self, interp_degree: int = 2, fs: int = 8000, n_bands: int = 8) -> np.ndarray:
        """
        Extrapolate absorption coefficients to fill the frequency spectrum from 0 to fs/2.
        The extrapolation is operated via fitting a polynomial of degree interp_degree to
        the set of absorption coefficients obtained from the database or measured, usually 
        covering a subset of (low) frequencies of the full spectrum. 
        
        Returns a vector of n_bands absorption coefficients. The coefficients correspond to
        frequency bands, with center frequencies equispaced in the range [0, fs/2].

        Parameters
        ----------
        interp_degree : int, optional
            Degree of polynomial used for interpolation, by default 2
        fs : int, optional
            Sampling frequency - used to define frequency spectrum, by default 8000
        n_bands : int, optional
            Number of bands of the spectrum from 0 Hz to fs/2 Hz, by default 8. Each band
            is defined by its center frequency, and a single absorption coefficient (absorption
            properties are assumed to be constant within the same band).

        Returns
        -------
        estimated_abs_coeffs: np.ndarray
            Array with n_bands entries, containing absorption coefficients for frequency range [0, fs/2] Hz
        
        Modifies
        --------
        absorption
            Parameter gets new extrapolated Absorption coefficients 
        reflection_coeffs
            Parameter gets new reflection coefficients computed from extrapolated absorption coefficients
        """
        
        coeffs = self.absorption["coeffs"]
        freqs = self.absorption["center_freqs"]
        
        # Transform in log-domain to perform interpolation
        log_freqs = np.log(freqs)
        log_coeffs = np.log(coeffs)
        log_model = np.polyfit(log_freqs, log_coeffs, interp_degree)
        
        # Define new frequency spectrum
        full_spectrum = np.linspace(0, 1, n_bands) * fs / 2
        full_spectrum[0] = 0.01   # Introduce small variation on first coeffs to perform polyval
        
        # Compute interpolated coeffs
        log_estimated_abs_coeffs = np.polyval(log_model, np.log(full_spectrum))
        estimated_abs_coeffs = np.exp(log_estimated_abs_coeffs)
        full_spectrum[0] = 0   # Restore original full_spectrum

        # Clip between 0 and 1
        estimated_abs_coeffs[estimated_abs_coeffs < 0.0] = 0.0
        estimated_abs_coeffs[estimated_abs_coeffs > 1.0] = 1.0

        self.absorption = {
            "description":  self.absorption["description"],
            "coeffs": estimated_abs_coeffs,
            "center_freqs": full_spectrum
        }
        
        # Compute reflection coefficients from absorption
        self.reflection_coeffs = np.sqrt(1 - estimated_abs_coeffs)

        return estimated_abs_coeffs

    def plot_absorption_coeffs(self) -> None:
        '''
        Plot the frequency values of the absorption coefficients of the material.
        '''

        plt.figure()
        plt.plot(self.absorption["center_freqs"], self.absorption["coeffs"])
        plt.title('Absorption Coefficients Road Surface')
        plt.ylabel(r'$\alpha$')
        plt.xlabel('f [Hz]')
        plt.show()