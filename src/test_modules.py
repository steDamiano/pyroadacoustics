import atmospheric_functions 
import interpolation_filters
import asphalt_absorption

import numpy as np

num_samples = 81
window = np.kaiser(num_samples, 0.4)

c = atmospheric_functions.compute_speed_sound(25)
filt = interpolation_filters.frac_delay_sinc(0.8, num_samples, window)
b_fir, theta_vector = asphalt_absorption.compute_angle_dependent_reflection_coeffs(11, 8000, 20)