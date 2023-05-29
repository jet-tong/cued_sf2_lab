import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
import numpy as np
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.dct import colxfm
from cued_sf2_lab.lbt import pot_ii
from cued_sf2_lab.dct import dct_ii
from cued_sf2_lab.laplacian_pyramid import quantise
from cued_sf2_lab.laplacian_pyramid import bpp
from cued_sf2_lab.dct import regroup
from scipy.optimize import minimize

def dctbpp(Yr, N):
    sum = 0
    # Your code here
    for i in range(N):
        for j in range(N):
            Ys = Yr[int(256/N)*i:int(256/N)*(i+1),int(256/N)*j:int(256/N)*(j+1)]
            sum += (256/N)*(256/N)*bpp(Ys)
    return sum

def DCT(X,N):
    cn = dct_ii(N)
    Y = colxfm(colxfm(X, cn).T,cn).T
    Z = colxfm(colxfm(Y.T, cn.T).T, cn.T)
    return Z


def encoder(X,N,):


