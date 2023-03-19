from src.core.CellMembraneKinetics import MembraneKinetics, MembraneCurrents
from src.core.FireNeuron import Neuron

mbk = MembraneKinetics()
t = mbk.t

Neuron = Neuron()
V, m, h, n, ina, ik, il = Neuron.generate_neuron_data(membrane_coef=mbk.membrane_coef)
Neuron.plot_signal(V, t, m, h, n, ina, ik, il)