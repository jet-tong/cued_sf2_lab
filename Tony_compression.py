import warnings
import inspect
import matplotlib.pyplot as plt
import IPython.display
import numpy as np
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.laplacian_pyramid import quantise
from cued_sf2_lab.laplacian_pyramid import bpp

from cued_sf2_lab.dct import dct_ii
from cued_sf2_lab.dct import regroup
from cued_sf2_lab.dct import colxfm
from cued_sf2_lab.lbt import pot_ii
from scipy.optimize import minimize

from math import floor, ceil
from sys import stdout as so
from bisect import bisect
from itertools import groupby



# Code for LBT: pot, dct, idct, then ipot.

def pot(X, N, s = 1):
    Xp = X.copy() # copy the non-transformed edges directly from X
    Xc = X.copy()

    t = np.s_[N//2:-N//2]
    pf,pr = pot_ii(N,s)
    cn = dct_ii(N)
    
    # First do XP =  PF * X
    Xp = colxfm(Xp[t,:],pf)
    Xp = colxfm(Xp[:,t].T, pf).T

    for i in range(256-N):
        for j in range(256-N):
            Xc[int(i+N/2),int(j+N/2)] = Xp[i,j]
    return Xc


def dct(X, N):
    cn = dct_ii(N)
    Y = colxfm(colxfm(X, cn).T, cn).T
    return Y

def dct_regroup(Y, N):
    Yr = regroup(Y, N)/N
    return Yr    

def dct_reshape(Yr, N = 8):
    # Reshape the image into a 4D array with shape (8, 32, 8, 32)
    Yrr = Yr.reshape(N, 256 // N, N, 256 // N) # TODO Variable N
    # Permute the axes to obtain the desired shape (8, 8, 32, 32)
    Yrr = Yrr.transpose(0, 2, 1, 3)
    return Yrr


def idct(Y, N):
    cn = dct_ii(N)
    Zp = colxfm(colxfm(Y.T, cn.T).T, cn.T)
    return Zp


def ipot(Zp, N, s = 1):
    t = np.s_[N//2:-N//2]
    pf,pr = pot_ii(N,s)
    cn = dct_ii(N)

    Z = Zp
    Z2 = Zp
    Z = colxfm(Z[:,t].T,pr.T).T
    Z = colxfm(Z[t,:],pr.T)
    for i in range(256-N):
        for j in range(256-N):
            Z2[int(i+N/2),int(j+N/2)] = Z[i,j]
    return Z2


def lbt(X, N=8, s=1.31, step=17, k=0.5): # rise1 = step/2 by default, so k=0.5 by default
    Xp = pot(X, N, s)
    Y = dct(Xp, N)
    return Y

def ilbt(Yq, N=8, s=1.31):
    Zpq = idct(Yq, N)
    Zq = ipot(Zpq, N, s)
    return Zq

def full_process_results(X, N=8, s=1.31, step=17, k=0.5): 
    Y = lbt(X, N, s, step, k)
    Yq = quantise(Y, step, rise1=k*step)
    Zq = ilbt(Yq, N, s)

    rms = np.std(Zq-X)
    bits_Yq = bpp(Yq) * Yq.size
    bits_X = bpp(X) * X.size
    cr = bits_X / bits_Yq  

    return rms, cr

def equal_bit_quantise(Y, bit):
    def objective1(step):
        Yq = quantise(Y,step, rise1=step)
        return np.abs(bpp(Yq) * Y.size - bit)
    result = minimize(objective1, 20, method='Nelder-Mead')
    final_step = result.x[0]
    Yq = quantise(Y, final_step)
    return Yq


def svdenc(idx,X):
    U,E,VT = np.linalg.svd(X)
    u1 = U[:,:idx]
    e1 = np.diag(E[:idx])
    v1t = VT[:idx,:]
    Y = np.concatenate((u1.T,e1,v1t),axis = 1)
    return Y

def svddec(idx,Y):
    u1 = Y[:,:256].T
    e1 = Y[:,256:256+49]
    v1t = Y[:,256+49:]
    Z = u1@e1@v1t
    return Z

def quantise_to_bits(Y, bit):
    def objective1(step):
        Yq = quantise(Y,step, rise1=step)
        vlc, header = encode(X,jpeg_quant_size = step)
        b = vlc[:,1].sum()
        return np.abs(b - bit)
    result = minimize(objective1, 20, method='Nelder-Mead')
    final_step = result.x[0]
    Yq = quantise(Y, final_step)
    return Yq