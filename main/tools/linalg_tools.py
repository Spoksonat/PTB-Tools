import numpy as np

# Special linear algebra functions

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