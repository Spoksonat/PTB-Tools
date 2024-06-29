import numpy as np

# Special functions for plotting

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


def conv3Dto2Dimage_V2(inMatrix3D, lScaleHorz, lScaleVert):

    lSize = inMatrix3D.shape

    out = np.zeros((lSize[0]*lScaleVert, lSize[1]*lScaleHorz))
    lZ  = 0
    
    for lH in range(0, lScaleHorz):
        for lV in range(0, lScaleVert):
            out[lV*lSize[0]:(lV+1)*lSize[0],lH*lSize[1]:(lH+1)*lSize[1]] = inMatrix3D[:,:,lZ]
            lZ += 1

    outMatrix2D = out

    return outMatrix2D

