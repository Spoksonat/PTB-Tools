import numpy as np
import mapvbvd
from skimage.segmentation import flood, flood_fill
import scipy.ndimage as ndimage
from skimage import measure
from skimage.filters import threshold_multiotsu
from datetime import datetime

"""
Image math functions

"""

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

"""
B1+ mapping functions

"""

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

"""
ROI functions

"""

def mask_region_growing(image, px, py, tol):
    image_mod = flood_fill(image, (int(py), int(px)), np.max(image)+1, tolerance=tol)
    mask = (image_mod == np.max(image)+1).astype(int)
    return mask

def mask_contour(image, threshold):
    contour = measure.find_contours(image, threshold)[0]
    mask = np.zeros(image.shape)
    mask[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1
    mask = ndimage.binary_fill_holes(mask)

    return contour, mask

def multi_otsu(b1rsos):
    
    images = np.zeros(b1rsos.shape)
    for i in range(b1rsos.shape[-1]):
       ths = threshold_multiotsu(b1rsos[:,:,i], classes=3)
       image = np.digitize(b1rsos[:,:,i], bins=ths)
       images[:,:,i] = image

    images[:,:,1] = np.sum(images[:,:,1:], axis=-1)
  
    return images

def normalize(image):
    return (image - image.min())/(image.max()-image.min())


"""
Linalg tools

"""

def makeColVec(inVec):

    vecsize = inVec.shape

    if vecsize[1] > vecsize[0]:
        colVec = np.swapaxes(inVec, axis1=0, axis2=1)
    else:
        colVec = inVec

    return colVec


def makeRowVec(inVec):

    vecsize = inVec.shape

    if vecsize[0] > vecsize[1]:
        rowVec = np.swapaxes(inVec, axis1=0, axis2=1)
    else:
        rowVec = inVec

    return rowVec

def multiprod(A, B):

    if len(np.shape(A)) == 2:
        return np.dot(A, B)

    #a = A.reshape(np.hstack([np.shape(A), [1]]))
    #b = B.reshape(np.hstack([[np.shape(B)[0]], [1], np.shape(B)[1:]]))
    #return np.sum(a*b, axis=2)
    return np.einsum('ijk,ikl->ijl', A, B)

def rot270(dimage):

    ds          = dimage.shape
    dRotImage   = np.zeros((ds[1],ds[0]))
    if len(ds) == 2:
        dRotImage = np.rot90(np.rot90(np.rot90(dimage)))
    elif len(ds) > 2:
        for lS in range(ds[2]):
            for lM in range(ds[3]):
                dRotImage[:,:,lS,lM] = np.rot90(np.rot90(np.rot90(dimage[:,:,lS,lM])))

    return dRotImage

def rot180(dimage):

    ds          = dimage.shape
    dRotImage   = np.zeros(ds)
    if len(ds) == 2:
        dRotImage = np.rot90(np.rot90(dimage))
    elif len(ds) > 2:
        for lS in range(ds[2]):
            for lM in range(ds[3]):
                dRotImage[:,:,lS,lM] = np.rot90(np.rot90(dimage[:,:,lS,lM]))

    return dRotImage

def rot90(dimage):

    ds          = dimage.shape
    dRotImage   = np.zeros((ds[1],ds[0]))
    if len(ds) == 2:
        dRotImage = np.rot90(dimage)
    elif len(ds) > 2:
        for lS in range(ds[2]):
            for lM in range(ds[3]):
                dRotImage[:,:,lS,lM] = np.rot90(dimage[:,:,lS,lM])

    return dRotImage

"""
Plot tools

"""

def conv3Dto2Dimage(inMatrix3D, lScaleHorz, lScaleVert):

    lSize = inMatrix3D.shape

    out = np.zeros((lSize[0]*lScaleVert, lSize[1]*lScaleHorz))
    lZ  = 0
    for lV in range(0, lScaleVert):
        for lH in range(0, lScaleHorz):
            out[lV*lSize[0]:(lV+1)*lSize[0],lH*lSize[1]:(lH+1)*lSize[1]] = inMatrix3D[:,:,lZ]
            lZ += 1

    outMatrix2D = out

    return outMatrix2D

""""
get tools

"""

def getDateTimeString():

    mClock              = datetime.now()
    DateAndTimeString   = mClock.strftime('%y%m%d_T%H%M%S')

    return DateAndTimeString, mClock

def getGamma(nucleus):
    switcher = {
        '1H': 267522209.9,
        '2H': 41.065*1e06,
        '3He': -203.789*1e06,
        '7Li': 103.962*1e06,
        '13C': 67.262*1e06,
        '14N': 19.331*1e06,
        '15N': -27.116*1e06,
        '17O': -36.264*1e06,
        '19F': 251.662*1e06,
        '23Na': 70.761*1e06,
        '31P': 108.291*1e06,
        '129Xe': -73.997*1e06
    }
    dGamma = switcher.get(nucleus, '1H')
    return dGamma

def getRectangularMask(roi, img_shape):

    x       = int(roi['x'][0])
    y       = int(roi['y'][0])
    width   = int(roi['width'][0])
    height  = int(roi['height'][0])

    mask = np.zeros(img_shape, dtype=bool)
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if x-width/2 <= i and i <= x+width/2 and y-height/2 <= j and j <= y+height/2:
                mask[i,j] = True

    return mask