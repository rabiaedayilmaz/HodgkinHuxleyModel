from flask import Flask, render_template, jsonify

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

# Define a route to display the data on a web page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/membrane_voltage')
def membrane_voltage():
    ### GRAPH PLOTS ###

    # membrane voltage
    trace0 = go.Scatter(
        x=df["time"],
        y=df["V"],
        name="Membrane Voltage",
    )

    plot_data=[trace0]

    graphJSON = json.dumps(plot_data, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('membrane_voltage.html', graphJSON=graphJSON)

@app.route('/membrane_currents')
def membrane_currents():
    ### GRAPH PLOTS ###

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

# Define a route to access the data as JSON
@app.route('/data')
def get_data():
    # Call your script to generate new data
    new_data = Neuron().generate_neuron_data(mbk.membrane_coef).tolist()

    # Update the data variable with the new data
    data.clear()
    data.extend(new_data)
    # Return the updated data as JSON
    return jsonify(data)