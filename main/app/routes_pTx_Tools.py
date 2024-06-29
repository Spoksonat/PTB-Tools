from flask import render_template
from app import app
from functions import B1plusmap_TXfct, TXsc_set_CV_as_CurrB1_map #, AFI_TXfct
import pickle
from bokeh.resources import INLINE
import tkinter as tk
from tools import prompt_tools

# This script contains the app routes for pTx tools menu. When clicking on a hyperlink (specified by href="") in the HTML, the corresponding function will be executed.
# The string defined in the @app.route() must be used for href in the HTML document.

@app.route('/pTx_Tools')
def pTx_Tools():
    return render_template('index.html')


if __name__ == '__init__':
    app.run(debug=True)