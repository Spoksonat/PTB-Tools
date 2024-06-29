import numpy as np
from datetime import datetime
from mahotas import polygon


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

def getMask(roi, img_shape):

    xs = roi['xs']
    ys = roi['ys']
    xs = [int(i) for i in xs]
    ys = [int(i) for i in ys]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    X = max_x-min_x+1
    Y = max_y-min_y+1

    newPoly = []
    for i in range(len(xs)):
        newPoly.append((xs[i]-min_x, ys[i]-min_y))

    grid    = np.zeros((X,Y), dtype=np.int64)
    polygon.fill_polygon(newPoly, grid)
    idx_i   = [i+min_x for i in np.where(grid)[0]]
    idx_j   = [j+min_y for j in np.where(grid)[1]]

    mask = np.zeros(img_shape, dtype=bool)
    mask[idx_i, idx_j] = True

    return mask

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