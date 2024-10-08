from flask import render_template, request
from ROI_TXfct import *
import pickle
import os
from bokeh.io import curdoc



content = ROI_class.Select_ROI_TXfct()

curdoc().add_root(content)