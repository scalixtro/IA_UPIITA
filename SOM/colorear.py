from skimage import io
import numpy as np

imagen2 = io.imread('imagen.jpg') / 255.0
imagen3 = np.zeros(imagen2.shape)

for ii in range(imagen2.shape[0]):
    for jj in range(imagen2.shape[1]):
        pixel = imagen2[ii, jj, :]
        columna = som.reshape(nfil * ncol, 3)
        d = 0

        for n in range(3):
            d += (pixel[n] - columna[:, n]) ** 2
        
        dist = np.sqrt(d)
        ind = np.argmin(dist)
        bmfil, bmcol = np.unravel_index(ind, [nfil, ncol])
        imagen3[ii, jj, :] = som[bmfil, bmcol, :]