import numpy as np

# Special Fourier transform functions

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
