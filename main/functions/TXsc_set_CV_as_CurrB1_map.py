import numpy as np


class TXsc_set_CV_as_CurrB1_map:
    def __init__(self, CVb1):
        self.cxmap      = CVb1.b1pcx
        self.type       = 'CV'
        self.scaling    = 1
        self.slices     = self.cxmap.shape[2]
        self.size       = self.cxmap.shape
        self.noofcha    = self.cxmap.shape[3]
        self.rotB1      = np.array([1,2,3])
        self.cxmap_orig = self.cxmap

class TXsc_set_CurrAFI_map:
    def __init__(self, AFImaps):
        self.cxmap      = AFImaps.afimap
        self.type       = 'AFI'
        self.scaling    = 1
        self.size       = self.cxmap.shape
        self.rotB1      = np.array([1,2,3])
        self.cxmap_orig = self.cxmap
