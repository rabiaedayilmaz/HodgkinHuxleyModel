from flask import Flask, render_template, jsonify, request

import plotly.graph_objects as go

import plotly
import pandas as pd
import json
import numpy as np

from src.core.FireNeuron import Neuron
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

app = Flask(__name__)

# Generate initial data
data = Neuron().generate_neuron_data(mbk.membrane_coef).tolist()

# extract data
V, m, h, n, ina, ik, il = data[0], data[1], data[2], data[3], data[4], data[5], data[6]

to_frame = {
    "time": np.arange(len(data[0])),
    "V": V,
    "m": m,
    "h": h,
    "n": n,
    "ina": ina,
    "ik": ik,
    "il": il,
}
# create data frame
df = pd.DataFrame(to_frame)
df2 = None

# Define a route to display the data on a web page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.form.get('apply') == 'APPLY':
            obv_time = request.args.get('obv_time')
            membrane_voltage = request.args.get('membrane_voltage')
            m = request.args.get('m')
            h = request.args.get('h')
            n = request.args.get('n')
            g_Na = request.args.get('g_Na')
            g_K = request.args.get('g_K')
            g_L = request.args.get('g_L')


            return render_template('index.html',
                                   membrane_voltage=membrane_voltage,
                                   obv_time=obv_time,
                                   m=m,
                                   h=h,
                                   n=n,
                                   g_Na=g_Na,
                                   g_K=g_K,
                                   g_L=g_L,
                                   )
    elif request.method == 'GET':
        return render_template('index.html')

@app.route('/membrane_voltage')
def membrane_voltage():
    ### GRAPH PLOTS ###

    if not df2 is None:
        trace0 = go.Scatter(
            x=df2["time"],
            y=df2["V"],
            name="Membrane Voltage",
        )
    else:
        # membrane voltage
        trace0 = go.Scatter(
            x=df["time"],
            y=df["V"],
            name="Membrane Voltage",
        )

    plot_data=[trace0]

    graphJSON = json.dumps(plot_data, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('membrane_voltage.html',
                           graphJSON=graphJSON,
                           )

@app.route('/membrane_currents')
def membrane_currents():
    ### GRAPH PLOTS ###

    if not df2 is None:
        # membrane currents
        trace1 = go.Scatter(
            x=df2["time"],
            y=df2["ina"],
            name="I_Na",
        )

        trace2 = go.Scatter(
            x=df2["time"],
            y=df2["ik"],
            name="I_K",
        )

        trace3 = go.Scatter(
            x=df2["time"],
            y=df2["il"],
            name="I_Leakage",
        )
    else:
        # membrane currents
        trace1 = go.Scatter(
            x=df["time"],
            y=df["ina"],
            name="I_Na",
        )

        trace2 = go.Scatter(
            x=df["time"],
            y=df["ik"],
            name="I_K",
        )

        trace3 = go.Scatter(
            x=df["time"],
            y=df["il"],
            name="I_Leakage",
        )
    plot_data1=[trace1, trace2, trace3]
    graphJSON1 = json.dumps(plot_data1, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('membrane_currents.html', graphJSON=graphJSON1)

@app.route('/gating_vars')
def gating_vars():
    ### GRAPH PLOTS ###

    if not df2 is None:
        # gating variables
        trace4 = go.Scatter(
            x=df2["time"],
            y=df2["m"],
            name="m",
        )

        trace5 = go.Scatter(
            x=df2["time"],
            y=df2["h"],
            name="h",
        )

        trace6 = go.Scatter(
            x=df2["time"],
            y=df2["n"],
            name="n",
        )
    else:
        # gating variables
        trace4 = go.Scatter(
            x=df["time"],
            y=df["m"],
            name="m",
        )

        trace5 = go.Scatter(
            x=df["time"],
            y=df["h"],
            name="h",
        )

        trace6 = go.Scatter(
            x=df["time"],
            y=df["n"],
            name="n",
        )

    plot_data2=[trace4, trace5, trace6]

    graphJSON2 = json.dumps(plot_data2, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('gating_vars.html', graphJSON=graphJSON2)

@app.route('/updated_params', methods=['POST'])
def update_params():
    global V_inp, t_inp, m_inp, h_inp, n_inp, gna_inp, gk_inp, gl_inp
    V_inp = float(request.form.get('membrane_voltage', False))
    t_inp = float(request.form.get('obv_time', False))
    m_inp = float(request.form.get('m', False))
    h_inp = float(request.form.get('h', False))
    n_inp = float(request.form.get('n', False))
    gna_inp = float(request.form.get('g_Na', False))
    gk_inp = float(request.form.get('g_K', False))
    gl_inp = float(request.form.get('g_L', False))

    mbk2 = MembraneKinetics(g_Na=gna_inp, g_K=gk_inp, g_L=gl_inp, membrane_voltage=V_inp, m=m_inp, h=h_inp, n=n_inp)
    mbc2 = MembraneCurrents(gna_inp, E_Na, gk_inp, E_K, gl_inp, E_L)
    data2 = Neuron().generate_neuron_data(mbk2.membrane_coef).tolist()
    #print(f'Parameters updated successfully!\n\nV: {V_inp}, t: {t_inp}, m: {m_inp}, h: {h_inp}, n: {n_inp}, ina: {gna_inp}, ik: {gk_inp}, il: {gl_inp}')
    global df2 # to access, shit, this looks really bad practice, need help to optimize

    # extract data
    V2, m2, h2, n2, ina2, ik2, il2 = data2[0], data2[1], data2[2], data2[3], data2[4], data2[5], data2[6]

    to_frame2 = {
        "time": np.arange(len(data2[0])),
        "V": V2,
        "m": m2,
        "h": h2,
        "n": n2,
        "ina": ina2,
        "ik": ik2,
        "il": il2,
    }
    # create data frame
    df2 = pd.DataFrame(to_frame2)

    return render_template('index.html', df2=df2.to_json())