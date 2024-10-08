import numpy as np
import tools as t
from functions import b1_phase_shimming_TXfct
import pickle
from pathlib import Path
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import LogColorMapper, ColorBar, LogTicker, HoverTool, Title
from bokeh.embed import components
from bokeh.layouts import column
from functions.Miscellaneous import *


class TXsc_B1PhaseShim:
    def __init__(self, optsdef):
        cxX0                    = np.load('/Users/manuelfernandosanchezalarcon/Desktop/PTB_Tools/main/files/Matlab_WS_Startphases_cxX0_1000x64.npy')
        self.optsdef            = optsdef
        self.optsdef['BETA0']   = cxX0

        # Open the saved B1p and ROI data
        with open('/Users/manuelfernandosanchezalarcon/Desktop/PTB_Tools/main/files/B1p_maps/B1p.pk1', 'rb') as input1:
            self.B1p = pickle.load(input1)
        #with open('/Users/manuelfernandosanchezalarcon/Desktop/PTB_Tools/main/files/ROIs/ROI.pk1', 'rb') as input2:
        #    self.ROI = pickle.load(input2)

        self.ROI = np.flipud(np.load("/Users/manuelfernandosanchezalarcon/Desktop/PTB_Tools/main/files/ROIs/ROI.npy"))

        shimming        = b1_phase_shimming_TXfct.b1_phase_shimming_TXfct(b1p=self.B1p.cxmap, roi=self.ROI, options=self.optsdef)

        self.betaAll    = shimming.betaAll
        self.fAll       = shimming.fAll
        print(self.betaAll)
        print(self.fAll)

        mmin        = self.fAll.min()
        mind        = self.fAll.argmin()
        lSolution   = mind

        self.CurSet = np.expand_dims(self.betaAll[:,lSolution], axis=1)
        self.ID     = getDateTimeString()
        self.Values = self.quantify_phase_shim_TXfct(idx=lSolution)

        mtmp = np.abs(np.dot(self.B1p.cxmap, makeColVec(self.CurSet)))
        self.B1pred_mag = np.abs(mtmp)
        self.B1pred_pha = np.angle(mtmp)

    def quantify_phase_shim_TXfct(self, idx):
        shimset = self.betaAll[:,idx]
        b1p     = self.B1p.cxmap
        roi     = self.ROI

        opts                = self.optsdef
        opts['TEST']        = 0
        opts['SAVEMAPS']    = 0

        shimvec     = makeColVec(np.expand_dims(shimset, axis=1))
        b1pat       = np.squeeze(np.dot(b1p, shimvec))
        b1sumofmag  = np.squeeze(np.dot(np.abs(b1p), np.abs(shimvec)))

        EfficiencyMap = np.divide(np.abs(b1pat), b1sumofmag)
        ValueStruct = {}
        ValueStruct['Efficiency']       = np.mean(EfficiencyMap[np.where(roi)])
        ValueStruct['EfficiencyMin']    = EfficiencyMap[np.where(roi)].min()
        ValueStruct['EfficiencyMax']    = EfficiencyMap[np.where(roi)].max()

        tmp = np.abs(b1pat[np.where(roi)])

        ValueStruct['CV'] = np.std(tmp)/np.mean(tmp)

        if opts['SAVEMAPS']:
            ValueStruct['B1Map']            = b1pat
            ValueStruct['EfficiencyMap']    = EfficiencyMap
        else:
            ValueStruct['B1Map']            = []
            ValueStruct['EfficiencyMap']    = []

        ValueStruct['ROIsize'] = np.sum(roi)

        return ValueStruct

    def show_shim_prediction_TXfct(self):
        opts = self.optsdef
        opts['TEST']        = 0
        opts['SAVEMAPS']    = 0
        opts['ROTATE']      = 0
        opts['COLORMAP']    = 'Inferno256'          #uses bokeh colormaps
        opts['COLORMAPEFF'] = 'Turbo256'

        shimvec = makeColVec(self.CurSet)

        b1pat_post  = np.abs(np.dot(self.B1p.cxmap, shimvec))
        b1pat_pre   = np.abs(np.dot(self.B1p.cxmap, np.ones(shimvec.shape)))
        b1pat_both  = np.concatenate((b1pat_pre, b1pat_post), axis=2)

        b1sumofmag  = np.dot(np.abs(self.B1p.cxmap), np.abs(shimvec))

        Eff_pre     = np.divide(np.abs(b1pat_pre), b1sumofmag)
        Eff_post    = np.divide(np.abs(b1pat_post), b1sumofmag)
        Eff_both    = np.concatenate((Eff_pre, Eff_post), axis=2)

        roi_both    = np.concatenate((self.ROI, self.ROI), axis=2)

        tmp_pre     = np.abs(b1pat_pre[np.where(self.ROI)])
        tmp_post    = np.abs(b1pat_post[np.where(self.ROI)])

        CV_pre  = np.std(tmp_pre)/np.mean(tmp_pre)
        CV_post = np.std(tmp_post)/np.mean(tmp_post)

        MeanEff_pre     = np.mean(Eff_pre[np.where(self.ROI)])
        MeanEff_post    = np.mean(Eff_post[np.where(self.ROI)])

        lNoOfSlices = self.B1p.cxmap.shape[2]

        if opts['ROTATE']==270:
            b1pat_both  = rot270(b1pat_both)
            roi_both    = rot270(roi_both)
            Eff_both    = rot270(Eff_both)
        elif opts['ROTATE']==180:
            b1pat_both  = rot180(b1pat_both)
            roi_both    = rot180(roi_both)
            Eff_both    = rot180(Eff_both)
        elif opts['ROTATE']==90:
            b1pat_both  = rot90(b1pat_both)
            roi_both    = rot90(roi_both)
            Eff_both    = rot90(Eff_both)

        sz = b1pat_both.shape

        data1   = {'image': [conv3Dto2Dimage(np.squeeze(b1pat_both), 2, lNoOfSlices)[::-1]]}
        src1    = ColumnDataSource(data1)
        data2   = {'image': [conv3Dto2Dimage(np.squeeze(Eff_both), 2, lNoOfSlices)[::-1]]}
        src2    = ColumnDataSource(data2)

        color_mapper1 = LogColorMapper(palette=opts['COLORMAP'])
        color_mapper2 = LogColorMapper(palette=opts['COLORMAPEFF'])

        fig1 = figure(x_range=(0,lNoOfSlices*sz[1]), y_range=(2*sz[0],0))
        fig1.border_fill_color = "#121212"
        fig1.background_fill_color = "#121212"
        fig1.title.text_color = "white"
        fig1.xaxis.major_label_text_color = "white"
        fig1.yaxis.major_label_text_color = "white"
        fig1.add_layout(Title(text='after shim (second column): CV = %f %%, mean efficiency: %f %%' %(CV_post*100, MeanEff_post*100)), 'above')
        fig1.add_layout(Title(text='before shim (first column): CV = %f %%, mean efficiency: %f %%' %(CV_pre*100, MeanEff_pre*100)), 'above')
        fig1.add_layout(Title(text='B1+ prediction'), 'above')
        fig1.image(image='image', x=0, y=0, dw=lNoOfSlices*sz[1], dh=2*sz[0], source=src1, color_mapper=color_mapper1)
        color_bar1 = ColorBar(color_mapper=color_mapper1, ticker=LogTicker(), label_standoff=12, location=(0, 0))
        color_bar1.background_fill_color = "#121212"
        color_bar1.major_label_text_color = "white"
        fig1.add_layout(color_bar1, 'right')
        fig1.add_tools(HoverTool(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")]))

        fig2 = figure(x_range=(0,lNoOfSlices*sz[1]), y_range=(2*sz[0],0))
        fig2.border_fill_color = "#121212"
        fig2.background_fill_color = "#121212"
        fig2.title.text_color = "white"
        fig2.xaxis.major_label_text_color = "white"
        fig2.yaxis.major_label_text_color = "white"
        fig2.add_layout(Title(text='after shim (second column): CV = %f %%, mean efficiency: %f %%' %(CV_post*100, MeanEff_post*100)), 'above')
        fig2.add_layout(Title(text='before shim (first column): CV = %f %%, mean efficiency: %f %%' %(CV_pre*100, MeanEff_pre*100)), 'above')
        fig2.add_layout(Title(text='B1+ prediction'), 'above')
        fig2.image(image='image', x=0, y=0, dw=lNoOfSlices*sz[1], dh=2*sz[0], source=src2, color_mapper=color_mapper2)
        color_bar2 = ColorBar(color_mapper=color_mapper2, ticker=LogTicker(), label_standoff=12, location=(0, 0))
        color_bar2.background_fill_color = "#121212"
        color_bar2.major_label_text_color = "white"
        fig2.add_layout(color_bar2, 'right')
        fig2.add_tools(HoverTool(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")]))

        script, div = components(column(fig1, fig2))

        return script, div

