import numpy as np
from functions.Miscellaneous import *
from scipy.optimize import minimize

class b1_phase_shimming_TXfct:
    def __init__(self, b1p, roi, options):

        self.optsdef                   = options
        self.optsdef['WHICHCHANNELS']  = np.arange(b1p.shape[3])
        self.optsdef['CHANNELVEC']     = np.append(self.optsdef['WHICHCHANNELS'], self.optsdef['WHICHCHANNELS'])

        b1p_        = removeInf(b1p)
        scfact      = 1

        b1psize     = b1p_.shape
        sz          = b1psize
        lNoOfCha    = b1psize[3]

        b1plustmp   = np.reshape(b1p_, (np.prod(sz[0:3]), sz[3]), order='F')        # matlab uses a different default ordering method than python when reshaping
        Atmp        = np.squeeze(b1plustmp[np.where(roi.ravel(order='F')),:])

        Ttmp        = roi.ravel()[np.where(roi.ravel())]
        lNonZeroInd = np.where(np.squeeze(Atmp@np.ones((lNoOfCha,1))))[0]
        self.A      = scfact*Atmp[lNonZeroInd,:]
        self.T      = Ttmp[lNonZeroInd]

        B                   = b1p_[np.where(self.optsdef['MASKSTAYSAME']), :]
        self.optsdef['B']   = B

        self.optsdef['LASTLOWINDEX'] = np.ceil(self.optsdef['SUMUPTOTHATTHRESHOLD'] * self.A.shape[0])

        self.betaAll = np.zeros((sz[3],self.optsdef['NOOFSTARTPHASES']), dtype=np.complex64)
        self.fAll    = np.zeros((self.optsdef['NOOFSTARTPHASES'], 1))

        
        for lL in range(self.optsdef['NOOFSTARTPHASES']):

            cxBeta0 = makeColVec(np.expand_dims(self.optsdef['BETA0'][lL,:sz[3]], axis=1))   # numpy squeezes arrays when you select elements like this.
            dBeta0  = np.concatenate((np.real(cxBeta0),np.imag(cxBeta0)), axis=0)                           # To use makeColVec the length of the shape has to be larger than 1
            
                                                                                                          # Therefore expand_dims must be used.
            x, fval = self.runOptimization(dX0=dBeta0)
            
            x       = np.expand_dims(x, axis=1)

            self.betaAll[:,lL]  = np.squeeze(makeColVec(x[:len(x)//2]+1j*x[len(x)//2:]))
            self.fAll[lL,0]     = fval


    def runOptimization(self, dX0):
        # I suggest to read the documentation of the minimize function: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
        # Other methods can be chosen. I just tested 'SLSQP'.
        #print('Separator \n')
        #print(dX0.reshape(len(dX0)))
        #print('Separator \n')
        switcher = {
            'homogeneity': minimize(self.objfunHomogeneity, dX0.reshape(len(dX0)), method='SLSQP',
                                    bounds=[(None, None) for x in range(len(dX0))],
                                    constraints=({'type':'eq', 'fun':self.ceq})
                                    ),
            'efficiency': minimize(self.objfunEfficiency, dX0.reshape(len(dX0)), method='SLSQP',
                                   bounds=[(None, None) for x in range(len(dX0))],
                                   constraints=({'type':'eq', 'fun':self.ceq})
                                   )
        }
        
        optim = switcher.get(self.optsdef['ALGO'].lower())
        print("x", optim.x)
        
        return optim.x, optim.fun

    def objfunHomogeneity(self, xx):

        A           = self.A
        dTVec       = self.T
        meanpower   = self.optsdef['MEANPOWER']

        xx   = np.expand_dims(xx, axis=1)                                       # the minimize function returns an array where the length of the shape is 1
        x    = makeColVec(xx[:len(xx)//2]+1j*xx[len(xx)//2:])
        a    = np.abs(A@x)

        meana   = np.mean(a)
        meant   = np.mean(dTVec)
        stda    = np.std(a-dTVec/meant*meana)
        fhom    = stda/(meana**meanpower)

        f = fhom

        return f

    def objfunEfficiency(self, xx):

        A       = self.A
        xx      = np.expand_dims(xx, axis=1)
        x       = makeColVec(xx[:len(xx)//2]+1j*xx[len(xx)//2:])
        feff    = -np.sum(np.divide(np.abs(A@x), np.abs(A)@np.abs(x)))

        f = feff

        return f

    def ceq(self, xx):
        x       = xx[:len(xx)//2]+1j*xx[len(xx)//2:]
        ceq     = np.abs(x)-1
        return ceq

    def cineq(self):
        cineq = []
        return cineq





