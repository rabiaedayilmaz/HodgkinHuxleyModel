import numpy as np

#########################################################
# Big thanks to @slarson                                #
# https://gist.github.com/slarson/37463b35ef8606629d2e  #
#########################################################

class MembraneKinetics:
    """ Implements kinetic equations for membrane """
    def __init__(self, C_m=1.0,
                 g_Na=120.0, g_K=36.0, g_L=0.3,
                 E_Na=50.0, E_K=-77.0, E_L=-54.387,
                 t=np.arange(0.0, 400.0, 0.1),
                 membrane_voltage=-65, m=0.05, h=0.6, n=0.32,
                 ):
        self.C_m = C_m
        self.g_Na = g_Na
        self.g_K = g_K
        self.g_L = g_L
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_L = E_L
        self.t = t
        self.membrane_coef = [membrane_voltage, m, h, n]

    # channel gating kinetics equations
    @staticmethod
    def alpha_m(V):
        return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))

    @staticmethod
    def beta_m(V):
        return 4.0 * np.exp(-(V + 65.0) / 18.0)

    @staticmethod
    def alpha_h(V):
        return 0.07 * np.exp(-(V + 65.0) / 20.0)

    @staticmethod
    def beta_h(V):
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    @staticmethod
    def alpha_n(V):
        return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))

    @staticmethod
    def beta_n(V):
        return 0.125 * np.exp(-(V + 65) / 80.0)


class MembraneCurrents:
    """ Implements current equations for membrane """
    def __init__(self, g_Na, E_Na, g_K, E_K, g_L, E_L):
        self.g_Na = g_Na
        self.E_Na = E_Na
        self.g_K = g_K
        self.E_K = E_K
        self.g_L = g_L
        self.E_L = E_L

    # Membrane current of Sodium
    def I_Na(self, V, m, h):
        return self.g_Na * m ** 3 * h * (V - self.E_Na)

    # Membrane current of Potassium
    def I_K(self, V, n):
        return self.g_K * n ** 4 * (V - self.E_K)

    # Membrane Leakage Current (because of Chlorine)
    def I_L(self, V):
        return self.g_L * (V - self.E_L)

    # External/Injected Current Applied to Membrane
    def I_inj(self, t):
        return 10 * (t > 100) - 10 * (t > 200) + 35 * (t > 300)
