import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from src.core.CellMembraneKinetics import MembraneKinetics, MembraneCurrents

mbk = MembraneKinetics()

###### MEMBRANE KINETICS VARIABLES ###########
C_m = mbk.C_m                                #
                                             #
g_Na = mbk.g_Na                              #
g_K = mbk.g_K                                #
g_L = mbk.g_L                                #
                                             #
E_Na = mbk.E_Na                              #
E_K = mbk.E_K                                #
E_L = mbk.E_L                                #
                                             #
t = mbk.t                                    #
                                             #
##############################################

mbc = MembraneCurrents(g_Na, E_Na, g_K, E_K, g_L, E_L)

def dALLdt(X, t):
    """
     X: An array contains [V, m, n, h]
     t: time range array

     V:         Voltage across membrane
     m and n:   controls Na+ channel
     h:         controls K+ gate

     m, n, h :  `gating variable`, probability that a channel is open at a given moment in time

     Since V, m, h, n are changes by time
     It calculates derivatives of V, m, h, and n
    """

    V, m, h, n = X  # extract values

    # calculate membrane potential & activation variables
    dVdt = (mbc.I_inj(t) - mbc.I_Na(V, m, h) - mbc.I_K(V, n) - mbc.I_L(V)) / C_m
    dmdt = mbk.alpha_m(V) * (1.0 - m) - mbk.beta_m(V) * m
    dhdt = mbk.alpha_h(V) * (1.0 - h) - mbk.beta_h(V) * h
    dndt = mbk.alpha_n(V) * (1.0 - n) - mbk.beta_n(V) * n

    return dVdt, dmdt, dhdt, dndt

class Neuron:
    """ Fires neuron and plot graphs """

    def __init__(self, membrane_coef=mbk.membrane_coef, t=mbk.t):
        self.membrane_coef = membrane_coef
        # calculate neuron data
        #V, m, h, n, ina, ik, il = self.generate_neuron_data(membrane_coef)
        # plot the signal
        #self.plot_signal(V, t, m, h, n, ina, ik, il)

    def generate_neuron_data(self, membrane_coef):
        """ Calculates the data to plot (voltage, currents, and gate probabilities) """

        X = odeint(dALLdt, membrane_coef, t)    # takes integral to find new values of V, m, h, n

        # extract values from array
        V = X[:, 0]
        m = X[:, 1]
        h = X[:, 2]
        n = X[:, 3]

        # calculate current for each channel Na+, K+, and leakage Cl-
        ina = mbc.I_Na(V, m, h)
        ik = mbc.I_K(V, n)
        il = mbc.I_L(V)

        return np.array([V, m, h, n, ina, ik, il])

    def plot_signal(self, V, t, m, h, n, ina, ik, il):
        """ Plot the currents, gating variables, injected current """

        plt.title('Hodgkin-Huxley Neuron')
        plt.plot(t, V, 'k')
        plt.ylabel('V (mV)')
        plt.show()

        plt.plot(t, ina, 'c', label='$I_{Na}$')
        plt.plot(t, ik, 'y', label='$I_{K}$')
        plt.plot(t, il, 'm', label='$I_{L}$')
        plt.ylabel('Current')
        plt.legend()
        plt.show()

        plt.plot(t, m, 'r', label='m')
        plt.plot(t, h, 'g', label='h')
        plt.plot(t, n, 'b', label='n')
        plt.ylabel('Gating Value')
        plt.legend()
        plt.show()

        plt.plot(t, mbc.I_inj(t), 'k')
        plt.xlabel('t (ms)')
        plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
        plt.ylim(-1, 31)
        plt.show()
