import numpy as np

def sound_attenuation_distance(d):
    ## Computes attenuation of sound at distance d from source:
    #   d: distance in m
    return 1 / (4 * np.pi * d)

def compute_air_impedance(T, p = 1, c = None):
    ## Computes air impedance given:
    #   T: temperature in Celsius
    #   p: atmospheric pressure in atm
    #   c: speed of sound in air, at temperature T
    T = T + 273.15
    if c == None:
        c = compute_speed_sound(T)
    p = p * 101325          # Convert atm to Pascal
    R_spec = 287.058
    return p / (R_spec * T) * c

def compute_speed_sound(T):
    ## Computes speed of sound in air at temperature T:
    #   T: temperature expressed in Celsius
    import numpy as np
    T = T + 273.15
    return 331.3 * np.sqrt(T / 273.15)

def compute_air_absorption_coeffs(T, p_s, rel_humidity, F):
    ## Computes the air absorption coefficients alpha:
    #   T: temperature in Celsius
    #   p_s: pressure in atm
    #   rel_humidity: relative humidity as a percentage
    #   F: frequencies to compute the coefficients for
    T = T + 273.15
    T_0 = 293.15        # Absolute atmospheric temperature
    p_s0 = 1            # Atmospheric Pressure in atm

    Csat = -6.8346 * (T_0 / T) ** 1.261 + 4.6151
    psat = 10 ** Csat
    H = H = psat * rel_humidity * p_s0 / p_s
    F_r0 = (p_s / p_s0) * (24 + 4.04 * 10**4 * H * (0.02 + H) / (0.391 + H))
    F_rN = (p_s / p_s0) * (T_0 / T)**(1/2) * ( 9 + 280 * H * np.exp(-4.17 * ((T_0 / T)**(1/3) - 1)))
    return 20 / np.log(10) * F ** 2 * (((1.84*10**(-11))/((T_0 / T)**(1/2) * p_s / p_s0)) +
            (T_0 / T)**(5/2) * ( (0.1068 * np.exp(-3352/T) * F_rN) / (F**2 + F_rN**2) +
            (0.01278*np.exp(-2239.1 / T) * F_r0) / (F**2 + F_r0**2)))
