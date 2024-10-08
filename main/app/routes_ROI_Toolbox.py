from flask import render_template, request
from app import app
from functions import ROI_TXfct
from bokeh.embed import server_document
from bokeh.server.server import Server
import pickle
from bokeh.resources import INLINE
import os
from tornado.ioloop import IOLoop
from threading import Thread

# routes for B1 Mapping
global optsdef_roi 
optsdef_roi = {}
optsdef_roi['COLORMAP']         = 'Inferno256'

# This script contains the app routes for the B1 mapping tool. When clicking on a hyperlink (specified by href="") in the HTML, the corresponding function will be executed.
# The string defined in the @app.route() must be used for href in the HTML document.

@app.route('/ROI_Toolbox')
def ROI_Toolbox():

    with open('/Users/manuelfernandosanchezalarcon/Desktop/PTB_Tools/main/temp/B1p.pk1', 'rb') as input:
        B1p = pickle.load(input)

    global ROI_class
    ROI_class       = ROI_TXfct.ROI_TXfct(B1p, optsdef_roi)

    return render_template('ROITool_Web.html')

@app.route('/ROI_Toolbox/selectROI/', methods=['GET'])
def Select_ROI():
    js_resources    = INLINE.render_js()
    css_resources   = INLINE.render_css()  
    #script, div     = ROI_class.Select_ROI_TXfct()
    data            = {"title": "ROI Selection"}

    #return render_template('plots.html', title='ROI Selection', script=script, div=div, js_resources=js_resources, css_resources=css_resources, data=data)

    def bk_worker():
        # Can't pass num_procs > 1 in this configuration. If you need to run multiple
        # processes, see e.g. flask_gunicorn_embed.py
        server = Server({'/selectROI': ROI_class.Select_ROI_TXfct}, io_loop=IOLoop(), allow_websocket_origin=["127.0.0.1:5000"])
        server.start()
        server.io_loop.start()

    Thread(target=bk_worker).start()

    script = server_document('http://localhost:5006/selectROI')
    return render_template('plots_ROI.html', title='ROI Selection', script=script, data=data)