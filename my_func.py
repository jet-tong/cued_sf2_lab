%matplotlib widget
import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
import numpy as np
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.laplacian_pyramid import rowdec
from cued_sf2_lab.laplacian_pyramid import rowint
from cued_sf2_lab.laplacian_pyramid import quantise
from cued_sf2_lab.laplacian_pyramid import bpp

# Tony's code
#----------------------------------------------------------------------------------------
def py4enc(X, h):
    X1 = rowdec(rowdec(X,h).T,h).T 
    Y0 = X - rowint(rowint(X1,2*h).T,2*h).T
    X2 = rowdec(rowdec(X1,h).T,h).T 
    Y1 = X1 - rowint(rowint(X2,2*h).T,2*h).T
    X3 = rowdec(rowdec(X2,h).T,h).T 
    Y2 = X2 - rowint(rowint(X3,2*h).T,2*h).T
    X4 = rowdec(rowdec(X3,h).T,h).T 
    Y3 = X3 - rowint(rowint(X4,2*h).T,2*h).T
    
    
    return Y0, Y1, Y2, Y3, X4


def py4dec(Y0, Y1, Y2, Y3, X4, h):
    # your code here
    Z3 = Y3 + rowint(rowint(X4,2*h).T,2*h).T
    Z2 = Y2 + rowint(rowint(Z3,2*h).T,2*h).T
    Z1 = Y1 + rowint(rowint(Z2,2*h).T,2*h).T
    Z0 = Y0 + rowint(rowint(Z1,2*h).T,2*h).T
    warnings.warn("you need to write this!")
    return Z3, Z2, Z1, Z0

def quantise_pyramid(Y0,Y1,Y2,Y3,X4,step):
    Y0q =quantise(Y0,step) 
    Y1q =quantise(Y1,step) 
    Y2q =quantise(Y2,step) 
    Y3q =quantise(Y3,step) 
    X4q =quantise(X4,step) 

    Z3q, Z2q, Z1q, Z0q = py4dec(Y0q, Y1q, Y2q, Y3q, X4q, h)
    return Z3q, Z2q, Z1q, Z0q

#Jensen's code-------------------------------------------------------------------------