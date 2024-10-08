from flask import render_template, request
from app import app
from functions import TXsc_B1PhaseShim
import pickle
from bokeh.resources import INLINE
import os
import numpy as np

# This script contains the app routes for the shimming tool. When clicking on a hyperlink (specified by href="") in the HTML, the corresponding function will be executed.
# The string defined in the @app.route() must be used for href in the HTML document.

global optsdef
optsdef = {}
optsdef['LAMBDA']               = 0
optsdef['MASKSTAYSAME']         = []
optsdef['NOOFSTARTPHASES']      = 8
optsdef['MEANPOWER']            = 1
optsdef['SUMUPTOTHATTHRESHOLD'] = 0.1
optsdef['EFFVALUEFORCE']        = 0.5

# This script contains the app routes for the B1 mapping tool. When clicking on a hyperlink (specified by href="") in the HTML, the corresponding function will be executed.
# The string defined in the @app.route() must be used for href in the HTML document.

@app.route('/B1_Shim_Toolbox')
def B1_Shimming_Toolbox():
    return render_template('B1ShimTool_Web.html')

@app.route('/B1Shim_Toolbox/HomShim/')
def optimize_B1p_homogeneity():
    global optsdef
    optsdef['ALGO'] = 'HOMOGENEITY'

    Shim = TXsc_B1PhaseShim.TXsc_B1PhaseShim(optsdef)
    
    # In the matlab version, this part is actually contained in the TXsc_B1PhaseShim function. It seemed more practical to me to place it here.
    ValuesAll = np.zeros(Shim.optsdef['NOOFSTARTPHASES'])
    for lL in range(Shim.optsdef['NOOFSTARTPHASES']):
        ValuesAll = Shim.quantify_phase_shim_TXfct(idx=lL)

    # ShimStr is then a list containing as much dictionaries as performed shims.
    global ShimStr
    ShimStr = []
    ShimStr.append({
        'BestShim': Shim.CurSet,
        'AllShims': Shim.betaAll,
        'ID': Shim.ID,
        'NoOfStartingPhases': Shim.optsdef['NOOFSTARTPHASES'],
        'Values': Shim.Values,
        'ValuesAll': ValuesAll
    })

    js_resources    = INLINE.render_js()
    css_resources   = INLINE.render_css()
    script, div     = Shim.show_shim_prediction_TXfct()
    data            = {"title": "\(B_1^+\) Homogeneity"}

    return render_template('plots.html', title='B1+ homogeneity', script=script, div=div, js_resources=js_resources, css_resources=css_resources, data=data)

@app.route('/B1Shim_Toolbox/EffShim/')
def optimize_B1p_efficiency():
    global optsdef
    optsdef['ALGO'] = 'EFFICIENCY'

    Shim = TXsc_B1PhaseShim.TXsc_B1PhaseShim(optsdef)

    ValuesAll = np.zeros(Shim.optsdef['NOOFSTARTPHASES'])
    for lL in range(Shim.optsdef['NOOFSTARTPHASES']):
        ValuesAll = Shim.quantify_phase_shim_TXfct(idx=lL)

    global ShimStr
    ShimStr.append({
        'BestShim': Shim.CurSet,
        'AllShims': Shim.betaAll,
        'ID': Shim.ID,
        'NoOfStartingPhases': Shim.optsdef['NOOFSTARTPHASES'],
        'Values': Shim.Values,
        'ValuesAll': ValuesAll
    })

    js_resources  = INLINE.render_js()
    css_resources = INLINE.render_css()
    script, div   = Shim.show_shim_prediction_TXfct()
    data          = {"title": "\(B_1^+\) Efficiency"}

    return render_template('plots.html', title='B1+ Efficiency', script=script, div=div, js_resources=js_resources, css_resources=css_resources, data=data)

@app.route('/B1_Shim_Toolbox/update_B1ShimmingOptions/')
def update_B1ShimmingOptions():
    pass



if __name__ == '__init__':
    app.run(debug=True)