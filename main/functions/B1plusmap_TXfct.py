import numpy as np
import tools as t
from tools import plot_tools
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import LogColorMapper, LinearColorMapper, ColorBar, LogTicker, BasicTicker, HoverTool
from bokeh.embed import components
from bokeh.layouts import column
from functions.Miscellaneous import *
import mapvbvd
from bokeh.models import LogColorMapper, LinearColorMapper, ColorBar, LogTicker, BasicTicker, HoverTool, ColumnDataSource, CustomJS, Div, TextInput, TapTool, Button, Slider

class B1plusmap_TXfct:
    def __init__(self, optsdef):
        self.optsdef    = optsdef
        self.rk         = None
        self.b1p_mag    = None
        self.b1p_pha    = None
        self.b1m_mag    = None
        self.b1m_pha    = None
        self.b1pcx      = None
        self.noise_mean = None
        self.noise_corr = None
        self.noofcha    = None
        self.pathname   = None
        self.filename   = None

        self.CVmap_read_TXfct()


    def CVmap_read_TXfct(self):
        opts = self.optsdef

        file_path   = opts['PATH_TO_RAW_FILE']

        image = load_data(opts, file_path)
        rk, b1p_mag, b1p_phase, b1m_mag, b1m_phase, noise_mean, noise_corr = B1mapping(opts, image)

        self.rk         = np.abs(rk)
        self.b1p_mag    = np.abs(b1p_mag)
        self.b1p_pha    = np.angle(b1p_phase)
        self.b1m_mag    = np.abs(b1m_mag)
        self.b1m_pha    = np.angle(b1m_phase)
        self.b1pcx      = self.b1p_mag * np.exp(1j * self.b1p_pha)
        self.noise_mean = noise_mean
        self.noise_corr = noise_corr
        self.noofcha    = self.b1pcx.shape[3]

    def Plot_B1_map_TXfct(self, pol):
        # input: pol string - 'plus' for the B1plus map and 'minus' for the B1minus map

        opts = self.optsdef
        Mag  = None
        Pha  = None

        if pol == 'plus':
            Mag = self.b1p_mag
            Pha = self.b1p_pha
        elif pol == 'minus':
            Mag = self.b1m_mag
            Pha = self.b1m_pha

        sz = Mag.shape

        lNoOfParShown   = 3
        lNoOfCha        = sz[3]
        lNoOfSli        = min(sz[2], lNoOfParShown)

        if False:#lNoOfSli == lNoOfParShown:
            mmag = np.moveaxis(Mag[:,:,round(sz[2]/(lNoOfParShown+1)):-1:round(sz[2]/(lNoOfParShown+1)),:], [0,1,2,3], [0,1,3,2])
            mpha = np.moveaxis(Pha[:,:,round(sz[2]/(lNoOfParShown+1)):-1:round(sz[2]/(lNoOfParShown+1)),:], [0,1,2,3], [0,1,3,2])
        else:
            mmag = np.moveaxis(Mag[:,:,:,:], [0,1,2,3], [0,1,3,2])
            mpha = np.moveaxis(Pha[:,:,:,:], [0,1,2,3], [0,1,3,2])

        mmag_reshaped = np.reshape(mmag, (sz[0],sz[1],sz[2]*sz[3]), order='F')
        mpha_reshaped = np.reshape(mpha, (sz[0],sz[1],sz[2]*sz[3]), order='F')

        x = sz[1] * lNoOfCha
        y = sz[0] * lNoOfSli

        rr = y/x

        # The bokeh package has the disadvantage, that the origin for image plots is in the lower left corner. Due to this, the image data has to be reversed.
        data_mag    = {'image': [conv3Dto2Dimage(mmag_reshaped, lNoOfCha, lNoOfSli)[::-1]]}    #[::-1] reverses the image data
        src_mag     = ColumnDataSource(data=data_mag)
        data_pha    = {'image': [abs(conv3Dto2Dimage(mpha_reshaped, lNoOfCha, lNoOfSli))[::-1]]}
        src_pha     = ColumnDataSource(data=data_pha)

        color_mapper1 = LogColorMapper(palette=opts['COLORMAP'])
        color_mapper2 = LinearColorMapper(palette='Greys256')

        fig1 = figure(width=opts['FIGWIDTH']*50, height=int(np.ceil(opts['FIGWIDTH']*rr))*5*lNoOfCha, x_range=(0,x), y_range=(y,0), title='Magnitude')
        fig1.border_fill_color = "#121212"
        fig1.background_fill_color = "#121212"
        fig1.title.text_color = "white"
        fig1.xaxis.major_label_text_color = "white"
        fig1.yaxis.major_label_text_color = "white"
        fig1.image(image='image', x=0, y=0, dw=x, dh=y, source=src_mag, color_mapper=color_mapper1)
        color_bar1 = ColorBar(color_mapper=color_mapper1, ticker=LogTicker(), label_standoff=12, location=(0,0))
        color_bar1.background_fill_color = "#121212"
        color_bar1.major_label_text_color = "white"
        fig1.add_layout(color_bar1, 'right')
        fig1.add_tools(HoverTool(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")]))

        fig2 = figure(width=opts['FIGWIDTH']*50, height=int(np.ceil(opts['FIGWIDTH']*rr))*5*lNoOfCha, x_range=(0,x), y_range=(y,0), title='Phase')
        fig2.border_fill_color = "#121212"
        fig2.background_fill_color = "#121212"
        fig2.title.text_color = "white"
        fig2.xaxis.major_label_text_color = "white"
        fig2.yaxis.major_label_text_color = "white"
        fig2.image(image='image', x=0, y=0, dw=x, dh=y, source=src_pha, color_mapper=color_mapper2)
        color_bar2 = ColorBar(color_mapper=color_mapper2, ticker=BasicTicker(), label_standoff=12, location=(0,0))
        color_bar2.background_fill_color = "#121212"
        color_bar2.major_label_text_color = "white"
        fig2.add_layout(color_bar2, 'right')
        fig2.add_tools(HoverTool(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")]))

        script, div = components(column(fig1, fig2))

        return script, div

    def show_noise_correlation_maps(self):

        sz      = self.noise_corr.shape
        maps    = np.multiply(np.abs(self.noise_corr), (np.ones(sz) - np.eye(sz[0])))
        lSize   = maps.shape
        if len(lSize) > 2:
            lScaleHorz = int(np.ceil(np.sqrt(lSize[2])))
            lScaleVert = int(np.ceil(lSize[2]/lScaleHorz))
        else:
            lScaleHorz = 1
            lScaleVert = 1
            maps       = np.expand_dims(maps, axis=2)

        data = {'image': [plot_tools.conv3Dto2Dimage(maps, lScaleHorz, lScaleVert)[::-1]]}
        src  = ColumnDataSource(data=data)
        color_mapper = LinearColorMapper(palette='Viridis256')
        fig = figure(width=500, height=500, x_range=(0,sz[0]), y_range=(sz[1],0))
        fig.border_fill_color = "#121212"
        fig.background_fill_color = "#121212"
        fig.title.text_color = "white"
        fig.xaxis.major_label_text_color = "white"
        fig.yaxis.major_label_text_color = "white"
        fig.image(image='image', x=0, y=0, dw=sz[0], dh=sz[1], source=src, color_mapper=color_mapper)
        color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(), label_standoff=12, location=(0,0))
        color_bar.background_fill_color = "#121212"
        color_bar.major_label_text_color = "white"
        fig.add_layout(color_bar, 'right')
        fig.add_tools(HoverTool(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")]))

        script, div = components(fig)

        return script, div        
    

class AFImap_TXfct:
    def __init__(self, optsdef):
        self.optsdef    = optsdef
        self.afimap     = None
        self.noofcha    = None
        self.pathname   = None
        self.filename   = None

        self.AFI_read_TXfct()

    def AFI_read_TXfct(self):
    
        """
        This program reconstructs AFI raw data. The reconstructed AFI maps are saved under
        the same name as a .npy file in the same directory. Additionally an AFI map will be
        shown for one slice of the image.
        The directory and filename of the raw data have to be specified (and eventually 
        the slice number).
        """
        # setup
        multiple_r_channels = True        

        # define directory and filename for the raw data (no dicom files) and the reconstruction directory
        #path_dir = r'Z:\_allgemein\projects\Niklas_pvp_afi\20230213_AFI_and_PEX\RAW'
        #filename = 'meas_MID51_ml_AFI_spoil1_117_5_rf_incr0_FID109327'

        opts = self.optsdef
        file_path   = opts['PATH_TO_RAW_FILE']    

        # Get the header (hdr) information
        twixObj = mapvbvd.mapVBVD(file_path)
        twixObj.image.flagRemoveOS = True
        kdata = twixObj.image['']        

        # Reorder the data to shape x,y,z,Rx,TR
        kdata = np.squeeze(np.moveaxis(kdata, 1, 3))        

        # Find out the repetition time (tr) in seconds that was used
        tr = twixObj.hdr.MeasYaps[('alTR', '0')] * 1e-6        

        # Find out the difference between the two TR in the AFI in seconds
        tr_diff = twixObj.search_header_for_val('MeasYaps', ('sWiPMemBlock', 'alFree',))[1] * 1e-6        

        # Calculate the n value of the AFI with n=tr2/tr1
        tr1 = tr - tr_diff
        tr2 = tr + tr_diff
        n = tr2 / tr1        

        # Reconstruct the images by applying a 3D fft
        kdata_shifted = np.fft.fftshift(kdata, axes=(0, 1, 2,))
        im = np.fft.ifftshift(
            np.fft.ifftn(kdata_shifted, (kdata_shifted.shape[0], kdata_shifted.shape[1], kdata_shifted.shape[2]), (0, 1, 2,),
                         norm=None), (0, 1, 2,))        

        # Calculate the sum of squares over the reception channels (third axis counting from zero) for each TR if         necessary
        if multiple_r_channels:
            im1 = np.sqrt(np.sum(abs(im[:, :, :, :, 0]) ** 2, axis=3))
            im2 = np.sqrt(np.sum(abs(im[:, :, :, :, 1]) ** 2, axis=3))
        else:
            im1 = abs(im[:, :, :, 0])
            im2 = abs(im[:, :, :, 1])        

        # Create a mask to deal with noise at places without signal. Be careful with the choice of percentage for         the threshold!
        mask = im1 > (0.05 * np.amax(im1))        

        # Calculate the signal ratios and mask them by setting low-signal-values to 1 for a flip angle of 0°
        r = im2 / im1
        r = r * mask
        r[r == 0] = 1  # Sets all values in array r to 1 that were set to zero by the mask        

        # Calculate the flip angle from the two 3D images with known TR1 and TR2 times
        AFI = 180 / np.pi * np.arccos((r * n - 1) / (n - r))

        self.afimap     = AFI

    def Plot_AFI_map_TXfct(self, doc):

        opts = self.optsdef

        sz      = self.afimap.shape

        slider_im = Slider(start=0, end=self.afimap.shape[-1]-1, value=0, step=1, title="Image index")
        maps    = self.afimap[:,:,slider_im.value]

        data = {'image': [maps]}
        src  = ColumnDataSource(data=data)
        color_mapper = LinearColorMapper(palette='Viridis256')

        def callback_change_image(attr, new, old):
           image_index = slider_im.value
           src.data = {"image": [self.afimap[:,:,image_index]]}

        
        fig = figure(width=500, height=500, x_range=(0,sz[0]), y_range=(sz[1],0))
        fig.border_fill_color = "#121212"
        fig.background_fill_color = "#121212"
        fig.title.text_color = "white"
        fig.xaxis.major_label_text_color = "white"
        fig.yaxis.major_label_text_color = "white"
        fig.image(image='image', x=0, y=0, dw=sz[0], dh=sz[1], source=src, color_mapper=color_mapper)
        color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(), label_standoff=12, location=(0,0))
        color_bar.background_fill_color = "#121212"
        color_bar.major_label_text_color = "white"
        fig.add_layout(color_bar, 'right')
        fig.add_tools(HoverTool(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")]))

        slider_im.on_change("value", callback_change_image)

        layout = column(fig, slider_im)

        doc.add_root(layout)      



