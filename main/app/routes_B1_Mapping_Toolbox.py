from flask import render_template, request
from app import app
from functions import B1plusmap_TXfct, TXsc_set_CV_as_CurrB1_map #, AFI_TXfct
import pickle
from bokeh.resources import INLINE
import os
from tornado.ioloop import IOLoop
from threading import Thread
from bokeh.server.server import Server
from bokeh.embed import server_document

# routes for B1 Mapping
global optsdef_b1map
optsdef_b1map = {}
optsdef_b1map['DIMB1']            = 2
optsdef_b1map['WHICHSLICES']      = 1
optsdef_b1map['FIGWIDTH']         = 20
optsdef_b1map['COLORMAP']         = 'Inferno256'    #see bokeh palettes for other colormaps
optsdef_b1map['RELPHASECHANNEL']  = 1
optsdef_b1map['USEMEAN']          = True
optsdef_b1map['SHOWMAPS']         = False
optsdef_b1map['B1PSCALING']       = 1
optsdef_b1map['BOOLEAN_LOAD']     = "false"

# This script contains the app routes for the B1 mapping tool. When clicking on a hyperlink (specified by href="") in the HTML, the corresponding function will be executed.
# The string defined in the @app.route() must be used for href in the HTML document.

@app.route('/B1_Mapping_Toolbox')
def B1_Mapping_Toolbox():
    full_path = os.path.abspath(__file__)
    path = full_path[:full_path.find("app")]
    optsdef_b1map['PATH_TO_MAIN']       = path
    data = {"boolLoadData" : optsdef_b1map['BOOLEAN_LOAD'] }
    return render_template('B1PlusMapping_Web.html', data=data)

@app.route('/process', methods=['POST']) 
def process(): 
    data = request.form.get('data') # String obtained from /process web path. The string is something like C:/fakepath/fileselected, but we want to obtain only the selected file 
    mode = request.form.get('loadMode')
    splitted_data = data.split(sep="\\")
    path_to_file = optsdef_b1map['PATH_TO_MAIN'] + "files/RAW/" + splitted_data[-1]
    print(path_to_file)
    optsdef_b1map['PATH_TO_RAW_FILE'] = path_to_file
    optsdef_b1map['LOAD_MODE'] = mode
    return path_to_file

@app.route('/process2', methods=['POST'])
def process2():
    name_b1p = request.form.get('name_b1p')
    optsdef_b1map['PATH_TO_SAVED_B1P'] = optsdef_b1map['PATH_TO_MAIN'] + "files/B1p_maps/" + name_b1p + ".pk1"
    return name_b1p

@app.route('/B1_Mapping_Toolbox/relative_B1Plus_Maps/')
def relative_B1Plus_Maps():
    if(optsdef_b1map['LOAD_MODE'] == "CV"):
        global CVb1
        CVb1 = B1plusmap_TXfct.B1plusmap_TXfct(optsdef_b1map)   
        # for later use, the B1 plus map will be saved as a .pk1 file (pickle file)
        B1p = TXsc_set_CV_as_CurrB1_map.TXsc_set_CV_as_CurrB1_map(CVb1)
        with open(optsdef_b1map['PATH_TO_MAIN'] + 'temp/B1p.pk1', 'wb') as output2:
            pickle.dump(B1p, output2, pickle.HIGHEST_PROTOCOL)
        optsdef_b1map['BOOLEAN_LOAD'] = "true" 
        data = {"boolLoadData" : optsdef_b1map['BOOLEAN_LOAD'] }
        return render_template('B1PlusMapping_Web.html', data=data)
    elif(optsdef_b1map['LOAD_MODE'] == "AFI"):
        global AFImaps
        AFImaps = B1plusmap_TXfct.AFImap_TXfct(optsdef_b1map)   
        # for later use, the B1 plus map will be saved as a .pk1 file (pickle file)
        AFI = TXsc_set_CV_as_CurrB1_map.TXsc_set_CurrAFI_map(AFImaps)
        with open(optsdef_b1map['PATH_TO_MAIN'] + 'temp/AFI.pk1', 'wb') as output2:
            pickle.dump(AFI, output2, pickle.HIGHEST_PROTOCOL)
        optsdef_b1map['BOOLEAN_LOAD'] = "true" 
        data = {"boolLoadData" : optsdef_b1map['BOOLEAN_LOAD'] }
        return render_template('B1PlusMapping_Web.html', data=data)
    else:
        raise TypeError("None is not a filetype")

@app.route('/B1_Mapping_Toolbox/b1p/')
def b1p():
    js_resources    = INLINE.render_js()
    css_resources   = INLINE.render_css()
    script, div     = CVb1.Plot_B1_map_TXfct(pol='plus')
    data            = {"title": "Relative \(B_1^+\) Maps"}

    return render_template('plots.html', title='B1+ Maps', script=script, div=div, js_resources=js_resources, css_resources=css_resources, data=data)

@app.route('/B1_Mapping_Toolbox/b1m/')
def b1m():
    js_resources    = INLINE.render_js()
    css_resources   = INLINE.render_css()
    script, div     = CVb1.Plot_B1_map_TXfct(pol='minus')
    data            = {"title": "Relative \(B_1^-\) Maps"}

    return render_template('plots.html', title='B1- Maps', script=script, div=div, js_resources=js_resources, css_resources=css_resources, data=data)

@app.route('/B1_Mapping_Toolbox/noise_correlation_maps/')
def noise_correlation_maps():
    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()
    script, div = CVb1.show_noise_correlation_maps()
    data            = {"title": "Noise Corr. Map"}

    return render_template('plots.html', title='Noise Corr. Maps', script=script, div=div, js_resources=js_resources, css_resources=css_resources, data=data)

@app.route('/B1_Mapping_Toolbox/save_b1p/')
def save_b1p():
    B1p = TXsc_set_CV_as_CurrB1_map.TXsc_set_CV_as_CurrB1_map(CVb1)
    with open(optsdef_b1map['PATH_TO_SAVED_B1P'], 'wb') as output2:
        pickle.dump(B1p, output2, pickle.HIGHEST_PROTOCOL)

    data = {"boolLoadData" : optsdef_b1map['BOOLEAN_LOAD'] }
    return render_template('B1PlusMapping_Web.html', data=data)

@app.route('/B1_Mapping_Toolbox/Plot_AFI/', methods=['GET'])
def plot_afi_all_slices():
    js_resources    = INLINE.render_js()
    css_resources   = INLINE.render_css()  
    data            = {"title": "AFI Maps"}

    def bk_worker():
        # Can't pass num_procs > 1 in this configuration. If you need to run multiple
        # processes, see e.g. flask_gunicorn_embed.py
        server = Server({'/selectROI': AFImaps.Plot_AFI_map_TXfct}, io_loop=IOLoop(), allow_websocket_origin=["127.0.0.1:5000"])
        server.start()
        server.io_loop.start()

    Thread(target=bk_worker).start()

    script = server_document('http://localhost:5006/selectROI')
    return render_template('plots_ROI.html', title='ROI Selection', script=script, data=data)




#routes for AFI reconstruction
optsdef_AFI = {}



if __name__ == '__init__':
    app.run(debug=True)