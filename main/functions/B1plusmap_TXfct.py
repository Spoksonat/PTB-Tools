import numpy as np
import tools as t
from tools import plot_tools
import mapvbvd
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import LogColorMapper, LinearColorMapper, ColorBar, LogTicker, BasicTicker, HoverTool
from bokeh.embed import components
from bokeh.layouts import column

def fftnsh(kspace, dims, bfftshift):

    lDims = len(dims)
    image = kspace

    if bfftshift:
        for lL in range(0, lDims):
            image = np.fft.fftshift(np.fft.fft(image, axis=dims[lL]), axes=dims[lL])
    else:
        for lL in range(0, lDims):
            image = np.fft.fft(image, axis=dims[lL])

    return image

def ifftnd(kspace, axes=[-1]):
    from numpy.fft import fftshift, ifftshift, ifftn
    if axes is None:
        axes = range(kspace.ndim)
    img = fftshift(ifftn(ifftshift(kspace, axes=axes), axes=axes), axes=axes)
    img *= np.sqrt(np.prod(np.take(img.shape, axes)))
    return img

def removeInf(inputMatrix):

    outputMatrix = inputMatrix

    outputMatrix = np.where(outputMatrix == np.inf, 0, outputMatrix)

    return outputMatrix

def removeNaN(inputMatrix):

    outputMatrix = inputMatrix

    outputMatrix = np.nan_to_num(outputMatrix, 0.0)

    return outputMatrix

def repmat(input_matrix, reps):
    len_reps = len(reps)
    len_input = len(input_matrix.shape)

    if len_input < len_reps:
        input_matrix_append = input_matrix
        while len(input_matrix_append.shape) < len_reps:
            input_matrix_append = np.reshape(input_matrix, (input_matrix_append.shape + (1,)))

        output_matrix = np.tile(input_matrix_append, reps)

    elif len_input == len_reps:
        output_matrix = np.tile(input_matrix, reps)

    elif len_input > len_reps:
        reps_append = reps
        while len(reps_append) < len_input:
            reps_append = reps_append + (1,)

        output_matrix = np.tile(input_matrix, reps_append)

    return output_matrix

def rms_comb(sig, axis=-1):
    return np.sqrt(np.sum(abs(sig)**2, axis))

def load_data(opts, file_path):
    twix_obj_B1                             = mapvbvd.mapVBVD(file_path) 
    twix_obj_B1.image.flagRemoveOS          = True
            
    #('Col', 'Cha', 'Lin', 'Ave', 'Sli', 'Par', 'Eco', 'Phs', 'Rep', 'Set', 'Seg', 'Ida', 'Idb', 'Idc', 'Idd', 'Ide')
    iindxs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    # We need: ('Col', 'Cha', 'Lin', 'Par', 'Sli', 'Ave', 'Phs', 'Eco', 'Rep', 'Set', 'Seg', 'Ida', 'Idb', 'Idc', 'Idd','Ide')
    iindxs_new = [0,1,2,5,4,3,7,6,8,9,10,11,12,13,14,15]
            
    rawsortTmp                               = np.moveaxis(twix_obj_B1.image[''],iindxs,iindxs_new)
    
    #mod csa 20200113: compute the sum if there are more than 1 segments        
    if rawsortTmp.ndim == 16:
        rawsortTmp = np.sum(rawsortTmp, axis=10)
    
    nDimsTmp    = np.arange(0, len(rawsortTmp.shape))
    nDims       = nDimsTmp[nDimsTmp != 1]

    lNonZeroElem = rawsortTmp
    for lD in nDims:
        lNonZeroElem = np.any(lNonZeroElem, axis=lD, keepdims=True)
    
    lNonZeroElem = np.nonzero(np.squeeze(lNonZeroElem))
    lNonZeroElem = lNonZeroElem[0]
    
    
    rawsort = rawsortTmp[:,lNonZeroElem,:,:,:,:,:,:]
    
    del nDimsTmp
    
    # perform fft
    if opts['DIMB1'] == 2:
        ima = np.moveaxis(fftnsh(np.fft.fftshift(np.fft.fftshift(rawsort, axes=0), axes=1), (0,2), True), [0,1,2,3,4,5,6,7], [0,3,1,5,2,6,7,4]) # previously [0,2,4,1,7,3,5,6] at the end
              
        #resort interleaved 2D slices
        if 'CV' in twix_obj_B1.hdr.Config.SequenceFileName:
            if ima.shape[2] > 1:
                imat = np.zeros((ima.shape), dtype=complex)
                if ima.shape[2] % 2:
                    imat[:, :, 1::2, :, :] = ima[:, :, 0:int(ima.shape[2]/2-0.5), :, :]
                    imat[:, :, 0::2, :, :] = ima[:, :, int(ima.shape[2]/2-0.5+1):, :, :]
                else:
                    imat[:, :, 1::2, :, :] = ima[:, :, 0:(ima.shape[2]//2), :, :]
                    imat[:, :, 0::2, :, :] = ima[:, :, (ima.shape[2]//2):, :, :]
                ima = imat
    elif opts['DIMB1'] == 3: #cw allow 3D datasets 7.10.2019
        ima = np.moveaxis(fftnsh(np.fft.fftshift(np.fft.fftshift(rawsort, axes=0), axes=1), (0,2,3), True), [0,1,2,3,4,5,67],     [0,3,1,2,5,6,7,4]) # previously [0,2,3,1,7,4,5,6] at the end
    
    """
    if(ima.shape[2] != 1): # Included on 19.10.2023
        ima = np.squeeze(ima)
    else: 
        ima = np.squeeze(ima, axis=(4,5,6,7,9,10,11,12,13,14))
    """
    ima = np.squeeze(ima)

    return ima

def B1mapping(opts, ima):
    sz  = ima.shape  
                
    noise_scan = ima[:,:,:,:,1]
    noise_mean = np.mean(np.abs(noise_scan)) / 1.253
    RX_sens = np.mean(np.mean(np.mean(np.abs(noise_scan), axis=0, keepdims=True), axis=1, keepdims=True), axis=2,    keepdims=True) / 1.253
    
    # calculate the correlation
    nn = np.reshape(noise_scan, (sz[0]*sz[1]*sz[2], sz[3]), order='F')
    
    noise_corr      = np.zeros((sz[3], sz[3]), dtype=complex)
    for lL in range(0, sz[3]):
        for lM in range(0, sz[3]):
            cc = np.corrcoef(nn[:,lL], nn[:,lM])
            noise_corr[lL,lM] = cc[1,0]
    
    # correct for different RX sensitivities
    #ima_cor = np.divide(ima[:,:,:,:,2:], t.MatRepmat.MatRepmat(RX_sens, (sz[0],sz[1],sz[2],1,sz[4]-2)))
    ima_cor = ima[:,:,:,:,2:]
    sz_cor  = ima_cor.shape
    
    # calculate the relative TX phase
    ima_cor_ref = np.sum(ima_cor, axis=4) / sz_cor[4]
    phasetemp   = removeInf(removeNaN(np.divide(ima_cor, repmat(ima_cor[:,:,:,:,opts['RELPHASECHANNEL']-1],(1,1,1,1,sz_cor[4])))))
    cxtemp      = np.sum(np.abs(ima_cor) * np.exp(1j * np.angle(phasetemp)), axis=3, keepdims=True)
    cxtemp2     = np.moveaxis(cxtemp, [0,1,2,3,4], [0,1,2,4,3])
    if (cxtemp2.shape[2]!= 1):
        cxtemp2     = np.squeeze(cxtemp2)
    else:
        cxtemp2     = np.squeeze(cxtemp2, axis=-1)
    b1p_phase   = np.exp(1j * np.angle(cxtemp2[:,:,:,:]))
    # calculate the TX magnitude
    if opts['USEMEAN']:
        # calculate as in ISMRM abstract
        imamag      = np.abs(ima_cor)
        b1_magtmp   = np.divide(np.sum(imamag, axis=3, keepdims=True), repmat(np.sum(np.sum(imamag, axis=3,keepdims=True),     axis=4, keepdims=True)**0.5,
                                                                                             (1,1,1,1,sz_cor[4])))
        b1p_mag     = np.moveaxis(b1_magtmp, [0,1,2,3,4], [0,1,2,4,3])
        print("b1p_mag size before", b1p_mag.shape)
        if (b1p_mag.shape[2]!= 1):
            b1p_mag     = np.squeeze(b1p_mag)
        else:
            b1p_mag     = np.squeeze(b1p_mag, axis=-1)
        print("b1p_mag size after", b1p_mag.shape)
        # calculate b1p_mag normalized with the CP mode
        sum_cp  = np.sqrt(np.sum(np.abs(np.sum(ima_cor, axis=4, keepdims=True)) ** 2, axis=3, keepdims=True))
        sum_cp  = np.squeeze(sum_cp)
        rk      = np.divide(np.sum(imamag, axis=3, keepdims=True), repmat(sum_cp, (1,1,1,1,sz_cor[4])))
        rk      = np.moveaxis(rk, [0,1,2,3,4], [0,1,2,4,3])
        rk      = np.squeeze(rk)
    else:
        # alternative calculation using median value
        imamag      = np.abs(ima_cor)
        b1_1        = np.divide(ima_cor, repmat(np.sum(np.abs(imamag), axis=4, keepdims=True), (1,1,1,1,sz_cor[4])))
        b1_2        = np.moveaxis(np.median(np.abs(b1_1), axis=3), [0,1,2,3,4], [0,1,2,4,3])
        b1p_magmed  = np.multiply(b1_2, repmat(np.squeeze(np.sum(np.sum(np.abs(imamag), axis=4, keepdims=True), axis=3,    keepdims=True)**0.5),
                                                              (1,1,1,sz_cor[4])))
    # calculate the relative RX phase
    ima_cor_tmp = ima_cor[:,:,:,opts['RELPHASECHANNEL']-1,:]
    ima_cor_tmp = ima_cor_tmp[:,:,:,np.newaxis,:]
    phasetemp   = removeInf(removeNaN(np.divide(ima_cor, repmat(ima_cor_tmp, (1,1,1,sz[3],1)))))
    cxtemp      = np.sum(np.abs(ima_cor) * np.exp(1j * np.angle(phasetemp)), axis=4, keepdims=True)
    
    if (cxtemp.shape[2]!= 1):
        cxtemp     = np.squeeze(cxtemp)
    else:
        cxtemp     = np.squeeze(cxtemp, axis=-1)
    b1m_phase   = np.exp(1j * np.angle(cxtemp[:,:,:,:]))
    # calculate TX magnitude
    b1m_mag = np.divide(np.sum(imamag, axis=4, keepdims=True), repmat(np.sum(np.sum(imamag, axis=3, keepdims=True),axis=4,     keepdims=True)**0.5, (1,1,1,sz[3],1)))
    if (b1m_mag.shape[2]!= 1):
        b1m_mag     = np.squeeze(b1m_mag)
    else:
        b1m_mag     = np.squeeze(b1m_mag, axis=-1)

    rk = np.rot90(rk, 1, axes=(0,1))
    b1p_mag = np.rot90(b1p_mag, 1, axes=(0,1))
    b1p_phase = np.rot90(b1p_phase, 1, axes=(0,1))
    b1m_mag = np.rot90(b1m_mag, 1, axes=(0,1))
    b1m_phase = np.rot90(b1m_phase, 1, axes=(0,1))

    return rk, b1p_mag, b1p_phase, b1m_mag, b1m_phase, noise_mean, noise_corr

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
        data_mag    = {'image': [t.plot_tools.conv3Dto2Dimage(mmag_reshaped, lNoOfCha, lNoOfSli)[::-1]]}    #[::-1] reverses the image data
        src_mag     = ColumnDataSource(data=data_mag)
        data_pha    = {'image': [abs(t.plot_tools.conv3Dto2Dimage(mpha_reshaped, lNoOfCha, lNoOfSli))[::-1]]}
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


