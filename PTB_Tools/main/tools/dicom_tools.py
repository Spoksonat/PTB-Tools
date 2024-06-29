import numpy as np
import os
import pydicom
from struct import Struct

# Functions to read dicom data.

def readDcmFolder(path):

    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.IMA' in file:
                files.append(os.path.join(r, file))

    sDcmInfo = []
    for f in files:
        ds = pydicom.dcmread(f)
        sDcmInfo.append(ds)

    sDcmsTmp            = np.zeros((sDcmInfo[0].Rows, sDcmInfo[0].Columns, len(sDcmInfo)))
    dSliceLocationTmp   = np.zeros((len(sDcmInfo),1))

    for i in range(len(sDcmInfo)):
        px                      = sDcmInfo[i].pixel_array
        sDcmsTmp[:,:,i]         = px
        dSliceLocationTmp[i]    = sDcmInfo[i].SliceLocation

    dSliceLocationTmp   = list(np.squeeze(dSliceLocationTmp))
    sortindex           = sorted(range(len(dSliceLocationTmp)), key=dSliceLocationTmp.__getitem__)

    sDcms = sDcmsTmp[:,:,sortindex]

    return  sDcms, sDcmInfo

# These functions and classes are copied from https://github.com/nipy/nibabel/tree/master/nibabel/nicom, because I had problems to install the nibabel package. This should
# be done to avoid copyright problems.

_ENDIAN_CODES = '@=<>!'

class Unpacker(object):
    def __init__(self, buf, ptr=0, endian=None):
        self.buf = buf
        self.ptr = ptr
        self.endian = endian
        self._cache = {}

    def unpack(self, fmt):
        pkst = self._cache.get(fmt)
        if pkst is None:
            if self.endian is None or fmt[0] in _ENDIAN_CODES:
                pkst = Struct(fmt)
            else:
                endian_fmt = self.endian + fmt
                pkst = Struct(endian_fmt)
                self._cache[endian_fmt] = pkst
            self._cache[fmt] = pkst
        values = pkst.unpack_from(self.buf, self.ptr)
        self.ptr += pkst.size
        return values

    def read(self, n_bytes=-1):
        start = self.ptr
        if n_bytes == -1:
            end = len(self.buf)
        else:
            end = start + n_bytes
        self.ptr = end
        return self.buf[start:end]


_CONVERTERS = {
    'FL': float,  # float
    'FD': float,  # double
    'DS': float,  # decimal string
    'SS': int,    # signed short
    'US': int,    # unsigned short
    'SL': int,    # signed long
    'UL': int,    # unsigned long
    'IS': int,    # integer string
}

MAX_CSA_ITEMS = 199

class CSAError(Exception):
    pass

class CSAReadError(CSAError):
    pass

def hdr_read(csa_str):
    csa_len = len(csa_str)
    csa_dict = {'tags': {}}
    hdr_id = csa_str[:4]
    up_str = Unpacker(csa_str, endian='<')
    if hdr_id == b'SV10':
        hdr_type = 2
        up_str.ptr = 4
        csa_dict['unused0'] = up_str.read(4)
    else:
        hdr_type = 1
    csa_dict['type'] = hdr_type
    csa_dict['n_tags'], csa_dict['check'] = up_str.unpack('2I')
    if not 0 < csa_dict['n_tags'] <= 128:
        raise CSAReadError('Number of tags `t` should be '
                           '0 < t <= 128')
    for tag_no in range(csa_dict['n_tags']):
        name, vm, vr, syngodt, n_items, last3 = \
            up_str.unpack('64si4s3i')
        vr = nt_str(vr)
        name = nt_str(name)
        tag = {'n_items': n_items,
               'vm': vm,
               'vr': vr,
               'syngodt': syngodt,
               'last3': last3,
               'tag_no': tag_no}
        if vm == 0:
            n_values = n_items
        else:
            n_values = vm

        converter = _CONVERTERS.get(vr)

        if tag_no == 1:
            tag0_n_items = n_items
        if n_items > MAX_CSA_ITEMS:
            raise CSAReadError('Expected <= {0} tags, got {1}'.format(
                MAX_CSA_ITEMS, n_items))
        items = []
        for item_no in range(n_items):
            x0, x1, x2, x3 = up_str.unpack('4i')
            ptr = up_str.ptr
            if hdr_type == 1:
                item_len = x0 - tag0_n_items
                if item_len < 0 or (ptr + item_len) > csa_len:
                    if item_no < vm:
                        items.append('')
                    break
            else:
                item_len = x1
                if (ptr + item_len) > csa_len:
                    raise CSAReadError('Item is too long, '
                                       'aborting read')
            if item_no >= n_values:
                assert item_len == 0
                continue
            item = nt_str(up_str.read(item_len))
            if converter:
                if item_len == 0:
                    n_values = item_no
                    continue
                item = converter(item)
            items.append(item)
            plus4 = item_len % 4
            if plus4 != 0:
                up_str.ptr += (4 - plus4)
        tag['items'] = items
        csa_dict['tags'][name] = tag
    return csa_dict

def nt_str(s):
    zero_pos = s.find(b'\x00')
    if zero_pos == -1:
        return s
    return s[:zero_pos].decode('latin-1')



