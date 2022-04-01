import numpy as np
import math

class DelayLine:
    """
    Class that defines a variable length Delay Line. A delay line applies a delay of M samples
    to the input signal that is being written on the delay line. When the value of M varies at each
    sampling instant, it generates a continuously varying delay line, i.e. a delay line where the
    distance between the read pointer and the write pointer is a function of time.

    In order to take into account a smooth change in the delay length M, the read operation must rely
    on an interpolation, whenever the position of the read pointer falls between two samples of the delay
    line (i.e. M is a non-integer number, and the delay is thus fractional). Different interpolation
    methods (linear, allpass, sinc) result in different levels of accuracy.

    The variable length Delay Line can be used to simulate sound propagation if the delay varies according
    to the physical laws governing sound propagation.

    The delay line is represented by a circular `ndarray` of N samples, with a single write pointer performing
    write operations (one write per sampling interval), and an arbitrary number of read pointers performing
    interpolated read operations (one operation per sampling interval for each pointer).

    Attributes
    ----------
    N: int
        Number of samples (i.e. entries) of the array defining the delay line
    delay_line: np.ndarray
        Circular array of N samples (i.e. entries) defining the delay line
    write_ptr: int
        Position of the write pointer that performs write operations on the delay line. Value must be in `[0, N-1]`
    read_ptr: float or np.ndarray
        Position of read pointers that perform read operations on the delay line. Each pointer value must be in
        `[0, N-1]`, and can take float values (interpolated read operations are performed)
    interpolation: str
        Interpolation method to be used for fractional delay reads. Can be 'Linear', 'Allpass' or 'Sinc'
    

    Methods
    -------
    set_delays(delay):
        Set initial read pointers position, as specified by the `delay` parameter (delay in number of samples)
    update_delay_line(x, delay):
        Writes signal value x on the delay line at the write_ptr position and increments write_ptr by 1.
        Performs interpolated read operation for each of the write pointers, and updates read pointers values
        based on the delay parameter (delay in number of samples)

    Notes
    -----
    A detailed description of the working principle of variable length delay lines is given in 
    
    `https://ccrma.stanford.edu/~jos/pasp/Variable_Delay_Lines.html`
    
    """

    def __init__(
            self, 
            N = 48000,
            num_read_ptrs = 1,
            interpolation = 'Allpass',
        ):
        """
        Creates a Delay Line object as a `ndarray` of N samples, with one write pointer and M read pointers.

        Parameters
        ----------
        N : int, optional
            Number of samples (i.e. elements) of array defining the delay line, by default 48000
        num_read_ptrs : int, optional
            Number of pointers that will read values from the delay line, by default 1. The `write_ptr` attribute
            will be instantiated as an 1D `ndarray` having num_read_pointers float entries.
        interpolation : str, optional
            String describing the type of interpolator to be used for fractional delay read operations. Can be:
            * Linear: linear interpolation
            * Sinc: sinc interpolation with filter length of 11 taps
            * Allpass: first order all pass interpolator
            
            By default: 'Allpass'
        
        Raises
        ------
        ValueError:
            If `N` is negative or equal to zero
        ValueError:
            If `num_read_ptrs` <= 0
        ValueError:
            If `interpolation` is neither Linear, nor Sinc, nor Allpass
        """

        if N <= 0:
            raise ValueError("Delay Line size must be greater than zero.")
        if num_read_ptrs <= 0:
            raise ValueError("Number of read pointers must be greater than zero.")
        if (interpolation != 'Linear' and interpolation != "Lagrange" and
            interpolation != 'Allpass' and interpolation != "Sinc"):
            raise ValueError("Interpolation parameter can be: `Linear`, `Allpass` or `Sinc`")

        self.N = N
        self.write_ptr = 0
        self.read_ptr = np.zeros(num_read_ptrs, dtype=float)
        self.delay_line = np.zeros(N)
        self.interpolation = interpolation

        self._SINC_SMP = 11
        self._WINDOW = np.hanning(self._SINC_SMP)
        _table_size = 51
        self._delta = 1 / 50
        self._table_delays = np.arange(0,1+self._delta,self._delta)
        
        n = np.arange(self._SINC_SMP)
        self._sinc_table = np.zeros((_table_size, self._SINC_SMP))

        for idx, val in enumerate(self._table_delays):
            self._sinc_table[idx] = np.sinc(n - (self._SINC_SMP - 1) / 2 - val)
            self._sinc_table[idx] *= self._WINDOW
            self._sinc_table[idx] = self._sinc_table[idx] / np.sum(self._sinc_table[idx])

        self._ya_alt = np.zeros(len(self.read_ptr))    # Initial buffer parameter for all-pass interpolation

    def set_delays(self, delay: np.ndarray) -> None:
        """
        Sets initial delays between the write pointer and each of the read pointers.

        Parameters
        ----------
        delay : np.ndarray
            1D Array with number of elements equal to number of read pointers of the delay line. Each entry 
            is the delay in number of samples between the write pointer and each of read pointers, and can
            take non-integer positive values.

        Raises
        ------
        ValueError:
            If one or more elements of the delay array is zero or negative
        ValueError:
            If len(delay) is different from number of read pointers
        RuntimeError:
            If one or more elements of the delay array are greater than the length of the delay line
        """

        if np.any(delay <= 0):
            raise ValueError('Delays must be non-negative numbers')
        if len(delay) != len(self.read_ptr):
            raise ValueError('Length of delay array should match number of read pointers')
        if np.any(delay >= self.N):
            raise RuntimeError('Delay greater than delay line length has been encountered. Consider'
            'using a longer delay line')
        
        for i in range(len(self.read_ptr)):
            self.read_ptr[i] = self.write_ptr - delay[i]
            # Check that read pointer value is in [0, N-1]
            if (self.read_ptr[i] < 0):
                self.read_ptr[i] += self.N

    def update_delay_line(self, x: float, delay: np.ndarray) -> np.ndarray:
        """
        Writes signal value x on the delay line at the write_ptr position and increments write_ptr by 1.
        Performs an interpolated read operation for each of the read pointers, and updates read pointer values
        based on the corresponding delay parameter (delay in number of samples).

        Parameters
        ----------
        x : float
            Input signal sample to be written on delay line at write pointer position
        delay : np.ndarray
            1D Array with number of elements equal to number of read pointers of the delay line. Each entry 
            is the delay in number of samples between the write pointer and each of read pointers, and can
            take non-integer positive values.

        Returns
        -------
        np.ndarray
            1D Array containing 1 interpolated sample per each of the read pointers of the delay line. These 
            values represent the output of the delay line (i.e. a delayed version of the input signal, interpolated
            to take into account fractional values of the delay)
        """
        
        # Create array to store output values
        y = np.zeros(len(self.read_ptr))
        

        # Append input sample at the write pointer position and increment write pointer
        self.delay_line[self.write_ptr] = x
        self.write_ptr += 1

        for i in range(len(self.read_ptr)):
            # Compute interpolated read position (fractional delay)
            rpi = int(self.read_ptr[i])
            frac_del = self.read_ptr[i] - rpi
        
            # Produce output with interpolated read
            y[i] = self._interpolated_read(rpi, frac_del, self.interpolation, i)

            # Update read pointer position
            self.read_ptr[i] = self.write_ptr - delay[i]
            # Check that read and write pointers are within delay line length
            while (self.read_ptr[i] < 0):
                self.read_ptr[i] += self.N
            while (self.write_ptr >= self.N):
                self.write_ptr -= self.N
            while (self.read_ptr[i] >= self.N):
                self.read_ptr[i] -= self.N
        return y

    def _interpolated_read(self, read_ptr_integer: int, d: float, method: str = 'Allpass', rptr_idx: int = 0) -> float:
        """
        Performs interpolated reads from delay line. Read pointer position is given to this function splitted into an
        integer part (`read_ptr_integer`) and a fractional part (`d`) in [0,1]. The interpolation is 
        performed considering neighbouring samples of the signal on the delay line and using the method
        specified in the `method` parameter.
        
        Parameters
        ----------
        read_ptr_integer : int
            Integer part of the delay
        d : float
            Fractional part of the delay, in the interval [0,1]
        method : str, optional
            Interpolation method used, by default 'Allpass'. Can be:
            * Linear: linear interpolation, uses 2 neighbouring samples
            * Lagrange: Lagrange polynomial interpolation, order 5 can be changed in this function. Implemented
            using an FIR filter
            * Sinc: sinc interpolation, implemented using windowed truncated sinc FIR filter with 11 taps. Window
            and number of taps can be modified in this function
            * Allpass: first order all pass interpolator, uses 2 neighbouring samples
        rptr_idx: int
            Index of the read pointer that is currently being used to perform the read. Needed to store the value of
            the previous sample for each pointer, in case allpass filter is used

        Returns
        -------
        float
            Sample computed after interpolated read, as a single float value

        Raises
        ------
        ValueError:
            If `method` is neither `Linear`, nor `Lagrange`, nor `Sinc`, nor `Allpass`
        """
        
        # TODO Vectorization to speed up convolution

        if method == 'Lagrange':
            # Compute interpolation filter
            order = 5
            h_lagrange = self._frac_delay_lagrange(order, d)
            
            # Convolve signal with filter
            out = 0
            for i in range(0, len(h_lagrange)):
                out = out + h_lagrange[i] * self.delay_line[(read_ptr_integer + i - math.floor(order/2)) % self.N]
            return out

        elif method == 'Linear':
            # Linear Interpolation formula
            return d * self.delay_line[(read_ptr_integer + 1) % self.N] + (1 - d) * self.delay_line[read_ptr_integer]
        
        elif method == 'Allpass':
            y = self.delay_line[read_ptr_integer] + d * (self.delay_line[(read_ptr_integer + 1) % self.N] 
                - self._ya_alt[rptr_idx])
            self._ya_alt[rptr_idx] = y
            return y

        elif method == 'Sinc':
            # Define windowed sinc filter paramters and compute coefficients
            h_sinc = self._frac_delay_interpolated_sinc(d)
            
            # Convolve signal with filter
            out = 0
            for i in range(0, self._SINC_SMP):
                out = out + h_sinc[i]*self.delay_line[(read_ptr_integer + i - math.floor(self._SINC_SMP/2)) % self.N]
            return out

        else:
            raise ValueError("Interpolation parameter can be: `Linear`, `Lagrange`, `Allpass` or `Sinc`")
    
    def _frac_delay_lagrange(self, order: int, delay: float) -> np.ndarray:
        """
        Computes Lagrange fractional delay FIR filter coefficients.

        Parameters
        ----------
        order : int
            Filter order. Must be an odd number, number of taps will be order + 1
        delay : float
            Fractional part of the delay

        Returns
        -------
        np.ndarray
            1D Array containing (order + 1) FIR filter coefficients
        """

        n = np.arange(0, order + 1)
        h = np.ones(order + 1)

        for k in range(0, order + 1):
            # Find index n != k
            index = []
            for j in range(0, order + 1):
                if j != k:
                    index.append(j)
            
            h[index] = h[index] * (delay - k) / (n[index] - k)
        return h

    def _frac_delay_interpolated_sinc(self, delay: float) -> np.ndarray:
        """
        Computes windowed sinc FIR filter coefficients for sinc interpolation, using a table lookup and linear
        interpolation. In the initialization of the `DelayLine` component, a sinc table is computed, containing
        the samples of windowed sinc filters corresponding to a predefined set of fractional delays between 0 and 1,
        with a step size equal to `self._delta`. 

        Given the fractional part of the delay `delay`, this functions retrieves the two sinc arrays corresponding to 
        the nearest available delay points, and performs a linear interpolation to compute the windowed sinc samples
        corresponding to the delay `delay`.

        Parameters
        ----------
        delay : float
            Fractional part of the delay, in [0,1]

        Returns
        -------
        np.ndarray
            Windowed-sinc filter coefficients corresponding to the delay `delay`
        """
        alpha = delay % self._delta
        position = np.searchsorted(self._table_delays, delay - alpha)
        if alpha < 1e-9:
            return self._sinc_table[position]
        else:
            return alpha * (self._sinc_table[position] - self._sinc_table[position + 1]) + self._sinc_table[position + 1]

    def _frac_delay_sinc(self, sinc_samples: int, sinc_window: np.ndarray, delay: float) -> np.ndarray:
        """
        Computes windowed sinc FIR filter coefficients for sinc interpolation.

        Parameters
        ----------
        sinc_samples : int
            Number of taps of the filter, should be an odd number
        sinc_window : np.ndarray
            Window to be applied to sinc function. Length of the window must be equal to sinc_samples
        delay : float
            Fractional part of the delay

        Returns
        -------
        np.ndarray
            1D array containing `sinc_samples` FIR filter coefficients
        """
        sinc_filter = sinc_window * np.sinc(np.arange(0,sinc_samples) - (sinc_samples - 1) / 2 - delay)
        
        return sinc_filter / np.sum(sinc_filter)
