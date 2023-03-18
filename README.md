# Hodgkin Huxley Model

## How to Use

coming soon idk

## Theory
### Part 1
The electrical properties of cells play a fundamental role in many biological processes. The membrane potential, which is the potential difference across the membrane is a critical aspect of physiology. Nernst equation and Goldman equation are widely used to understand and calculate resting membrane potential and movement of ions in the membrane. The Nernst equation describes the equilibrium potential for a single ion based on its concentration gradient and the membrane potential. On the other hand, the Goldman equation considers the permeability of the membrane for different ions and their respective concentration gradients. In this study, the fundamentals of membrane potential, the Nernst equation, and the Goldman equation will be described.

Electricity or electrical current can not be generated by cells, although all cells -not only neurons- have membrane resting potential. The thing traveling across cell is not electrons but waves of rapidly changing ionic charge differences. The difference in electrical charge between inside and outside of the neuron is the membrane potential. This can be measured by using two electrodes: a reference and a recording electrode. The reference electrode is placed in the extracellular solution which means body fluid that is not contained in cells. The recording electrode is inserted into the cell body of the neuron.

![Recording Potentials of Neuron](/assets/recording_neuron.png/)[^1]

**Equilibrium Potential:**  It is defined as the membrane potential of a neuron when the ion is in a state of balance. At this point, there is no net movement of the ion in either direction. Specifically, during the resting state of the neuron, the inside of the neuron is more negatively charged than the outside, which generates an electrical gradient that encourages the movement of ions. When the sodium channels are activated, the neuron's membrane becomes permeable to sodium, and sodium ions begin to flow across the membrane, their direction being influenced by electrochemical gradients. In the extracellular solution, the concentration of sodium is about ten times higher than in the intracellular solution, creating a concentration gradient that drives sodium into the cell. As sodium moves into the cell, the driving forces of these gradients change. When the membrane potential of the neuron becomes positive, the electrical gradient no longer favors the influx of sodium ions. Ultimately, the concentration gradient driving sodium into the neuron and the electrical gradient driving sodium out of the neuron become balanced with equal and opposite strengths, and the system reaches equilibrium [^2].

**Nernst Equation:**  Since the gradients acting on the ion will always drive the ion towards equilibrium, equilibrium potential can be calculated via Nernst Equation. This equation is derived from Gibbs free energy under standard conditions.

$$ E^0=E_{reduction}^0-E_{oxidation}^0\ \ \ \ (1)

ΔG is also related to E under general conditions (standard or not), where n is the number of electrons transferred in the reaction from balanced reaction, F is the Faraday constant (96,500 C/mol), and E is potential difference.

$$ ∆G=-nFE\ \ \ \ (2)

Under standard a condition, Equation 2 can be described as following.

$$ ∆G0=-nFE{0}\ \ \ \ (3)

According to Equation 3, when E0 is positive, the reaction is spontaneous and when E0 is negative, the reaction is non-spontaneous. From the thermodynamics, the Gibbs energy change under non-standard conditions can be expressed using Gibbs energy under standard conditions equation.

$$ ∆G = ∆G{0} + RTlnQ\ \ \ \ (4)

By substituting $∆G{0} expression with Equation 3 and $∆G expression with Equation 4, the overall equation turns into this.

$$ -nFE\ =\ {-nFE}^0\ +\ RT\ln{Q\ \ \ \ (5)}

By dividing both side of the Equation 5 with nF, this will be obtained.

$$ E\ =\ E^0\ -\ \frac{RT}{nF}\ln{Q\ \ \ \ (6)}

The $lnQ expression can be converted to base 10 from e. This conversion takes reference from the common chemistry equation.

$$ \ln{x\ =\ 2.303\times\ \log_{10}{x\ \ \ \ (7)}}

To prove the Equation 7, substituting x by 10 is enough as ln10\ =\ 2.303\times1 is obtained that is true and consistent. By using the property of Equation 7 and applying it to Equation 6, this simplified equation is obtained.

$$ E\ =\ E^0\ -\ \frac{2.303RT}{nF}\log_{10}{Q\ \ \ \ (8)}

At standard temperature (25OC) T equals 298OK. On the other hand, R and F are constants. Thus, for further simplification $\frac{RT}{F} can be written as 0.0592 V. Finally, Equation 8 can be written as this.

$$ E\ =\ E^0\ -\ \frac{0.0592V}{n}\log_{10}{Q\ \ \ \ (9)}

**Goldman-Hodgkin-Katz Equation:**  While the Nernst Equation provides a quantitative measure of equality that exists between chemical and electrical gradients for individual ions, Goldman Equation calculates the predicted resting membrane potential that reflects the relative contributions of the chemical concentration gradients and relative membrane permeability for a multitude ions collectively [^3]. 
It is described as following, where Em is the membrane potential, R is the universal gas constant, T is the temperature in Kelvin, F is the Faraday constant, pM is the permeability for M+/- and [M+/-] is the concentration of M+/- for input(in) or output(out).

$$ E_m=\frac{RT}{F}\ln{\left(\frac{\sum_{i}^{n}{P_{M_i^+}\left[M_i^+\right]_{out}+}\sum_{j}^{m}{P_j^-\left[A_j^-\right]_{in}}}{\sum_{i}^{n}{P_{M_i^+}\left[M_i^+\right]_{in}+}\sum_{j}^{m}{P_j^-\left[A_j^-\right]_{out}}}\right)}\ \ \ \ (10)

It is a resonance the Nernst Equation but has a term for each permeant ion. For instance, to calculate Na+, K+ and Cl- membrane potentials, the Equation 10 will be taken as template, and final equation can be written and calculated.

$$ E_m=\frac{RT}{F}\ln{\left(\frac{\sum_{i}^{n}{P_{K^+}\left[K_i^+\right]_{out}+\sum_{k}^{l}{P_{{Na}^+}\left[{Na}_k^+\right]_{out}}+}\sum_{j}^{m}{P_{{Cl}^-}^-\left[{Cl}_j^-\right]_{in}}}{\sum_{i}^{n}{P_{K^+}\left[K_i^+\right]_{in}+\sum_{k}^{l}{P_{{Na}^+}\left[{Na}_k^+\right]_{in}}+}\sum_{j}^{m}{P_{{Cl}^-}^-\left[{Cl}_j^-\right]_{out}}}\right)}\ \ \ \ (11)

As a result, resting membrane potential is determined by the balance of ion concentrations inside and outside of the cell. The Nernst equation is derived step by step and used to calculate the equilibrium potential for individual ions. However, for cells that are permeable to multiple ions, the Goldman equation is required to accurately calculate the resting membrane potential. Overall, it is concluded that to calculate and analyze the potential membrane equilibrium of cells, Nernst and Goldman equations are powerful tools.

**REFERENCES**

[^1] Baek, K. (2018, May 31). Introduction to Electrophysiology. https://gate.nmr.mgh.harvard.edu/wiki/whynhow/images/e/eb/WhyNHow-Electrophysiology.pdf

[^2] Henley, C. (2021). Membrane Potential. In Pressbooks. https://openbooks.lib.msu.edu/neuroscience/chapter/membrane-potential/

[^3] Gilbert-Kawai, E., & Wittenberg, M. (2014). Goldman equation. Cambridge University Press EBooks, 149–150. https://doi.org/10.1017/cbo9781139565387.077


### Part 2
A system is formed by a group of interacting elements that are affiliated to a set of rules to obtain desired or reasonable output for a given input. Modeling is an essential tool and simulation is a crucial test mechanism for engineers and scientists. 
Hodgkin-Huxley (HH) [^1] conducted experiments on the giant axon and found three different types of ion current that are sodium, potassium, and a leak current that consists of {Cl}^-\ ions. The core of mathematical framework of this model was developed around 1950s by Alan Hodgkin and Andrew Huxley [^2]. They demonstrated that there is a flow of those ions through the cell membrane and specific voltage dependent ion channels, one is for sodium and another one is for potassium, controls this flow. 

The semipermeable cell membrane separates the interior of the cell
from the extracellular liquid and membrane acts as a capacitor. Moreover, if an input 
current I(t) is injected into the cell; it may cause further charge on the capacitor or leak 
through the channels in the cell membrane. The cell membrane is electrically modeled in 
Figure 1. The resistor, in Figure 1, represents each channel type. Unspecific channel has a 
leak resistance, RL; the sodium channel has a resistance RK; and potassium channel has a 
resistance RNa. The diagonal arrow on the capacitors indicates that the value of capacitance 
is not fixed. Because the capacitance changes with respect to the ion channel is open or 
closed, since the ion concentration is different between inside of the cell and extracellular 
liquid due to the active ion transport through the cell membrane. The Nernst potential 
generated by the difference ion concentration is represented by a battery. However, Nernst 
potential is different for each ion type. Thus, there are separate batteries for sodium, 
potassium, and unspecific channel, with voltages ENa, EK, and EL, respectively.

![Recording Potentials of Neuron](/assets/circuit_hh_model.png) [^3]

The conversation of electrical charge on the membrane is described as:

$$ I(t)=\ I_c(t)+\sum_{k}{I_k(t)}\ \ \ \ (1)

Equation (1) implies the applied current I(t) can be split in a capacitive IC that charges the capacitor C  and moreover components I_{k\ }that pass through ion channels where summation contains all ion channels. From the equation of $C\ =\ \frac{q}{u} where q is the charge and u is the voltage across the capacitor, the charging current can be written as $I_c(t)\ =\ C\frac{du}{dt}. From there, applying to Equation 1, it is obtained:

$$ C\frac{du}{dt}=-\sum_{k}{I_k(t)+I(t)}\ \ \ \ (2)

Overall, in biological terms, u is the voltage across the membrane and -\sum_{k}{I_k(t)} is the summation of the ionic currents that pass through the cell membrane.
If Kirchhoff’s Current Law, that states summation of currents equals zero, is applied to Figure 1, while assuming total voltage across membrane that is between inside and outside nodes Vm and summation of all currents across membrane (INa, IK, IL) is Im, firstly this KCL equation is obtained:

$$ I_{Na}+I_K+I_L+I_C-I_{ext}=0 \ \ \ \ (3)

Stating this equation for membrane:

$$ I_C=\ C_m\times\sfrac{dV_m}{dt} \ \ \ \ (4)

By substituting Equation 4 to Equation 3, finally obtained:

$$ \sfrac{{dV}_m}{dt}=\ \sfrac{(I_{ext}-I_{Na}-I_K-I_L)}{C_m} \ \ \ \ (5)

The voltage-independent conductance of leakage channel is $g_L=\sfrac{1}{R_L}, the voltage at the leak resistor is $(u-E_L), and by Ohm’s Law the leak current is obtained as $I_L=g_L\left(u-E_L\right).
If all channels are open, they transmit currents with a maximum conductance $g_{Na} and $g_{K}, respectively. However, normally some channel will be blocked. The advancement of HH model is that it demonstrates us how the effective resistance of a channel changes as a function of time and voltage. They also introduced gating variables, m, n, and h to model that are the probability of that a channel is open at a given moment in time. The combination of m and h controls the ${Na}^+channels and $K^+gates are controlled by n. The conductance of sodium channels can be modeled as $\sfrac{1}{R_{Na}=g_{Na}m^3h} while m is the activation of the channel and h is inactivation. The conductance of potassium is $\sfrac{1}{R_K=g_Kn^4} while n is the activation of the channel. 
Finally, Hodgkin and Huxley formulated the three ion currents on the Equation 2 and obtained:

$$ \sum_{k}{I_K=g_{Na}m^3h\left(u-E_{Na}\right)+g_Kn^4\left(u-E_K\right)+g_L\left(u-E_L\right)} \ \ \ \ (6)

The Hodgkin-Huxley is a fundamental model in neuroscience and provides insight into mechanism of action potentials in neurons.

**REFERENCES**
[1], [2] Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. The Journal of Physiology, 117(4), 500–544. https://doi.org/10.1113/jphysiol.1952.sp004764

[3] Mertens, D. R. (1977). Principles of Modeling and Simulation in Teaching and Research. Journal of Dairy Science, 60(7), 1176–1186. https://doi.org/10.3168/jds.s0022-0302(77)84005-8

