""" This file contains the `encode` function. Feel free to split it into smaller functions """
import numpy as np
from typing import Tuple, Any
from cued_sf2_lab.jpeg import jpegenc
from jteg import *
import math


from common import HeaderType, jpeg_quant_size


from scipy.optimize import minimize_scalar

def compress_to_bit(X, required_bits = 40960-1424-24, s=1.085):
    def objective(quant_size):
        #vlc, _ = encode(X, quant_size, s=s)
        vlc, _ = lbtenc(X-128.0, quant_size, opthuff=True, log=False, s = s)
        bits = vlc[:, 1].sum()
        #return abs(bits - required_bits)
        if bits > required_bits:
            return np.abs(bits - (required_bits))**2 * 10  # Soft penalty, preserving the gradient
            #f = np.abs(bits-required_bits)
            #p = 0.001
            #B = 1 / np.abs(bits-required_bits) + (bits > required_bits)*20 # Barrier Function
            #return f - p / B
        else:
            return np.abs(bits - required_bits)

    result = minimize_scalar(objective, bounds=(5, 500), method='bounded')
    quant_size_final = math.ceil(result.x * 1000) / 1000  # Round up to 3 decimal places
    #quant_size_final = result.x
    vlc, hufftab = lbtenc(X-128.0, quant_size_final, opthuff=True, log=False, s = s)
    return vlc, hufftab, quant_size_final

def header_bits(header: HeaderType) -> int:
    """ Estimate the number of bits in your header.
    
    If you have no header, return `0`. """
    # replace this with your size estimate, and a comment explaining how you got it!
    # Transfer the hufftab (1424), quant_size (8) and LBT's s (16)

    return (len(header.bits) + len(header.huffval)) * 8 + 8 + 16



def encode(X: np.ndarray, jpeg_quant_size : float = 48.115, s : float = 1.6099190603156157) -> Tuple[np.ndarray, HeaderType]:
    """
    Parameters:
        X: the input grayscale image
    
    Outputs:
        vlc: the variable-length codes
        header: any additional parameters to be saved alongside the image
    """
    # replace this with your chosen encoding scheme. If you do not use a header,
    # then `return vlc, None`.
    X = X - 128.0
    #vlc, hufftab = lbtenc(X, jpeg_quant_size, opthuff=True, log=False, s=s)
    #vlc, hufftab = jpegenc(X, jpeg_quant_size, opthuff=True, log=False)

    vlc, hufftab, quant_size_final = compress_to_bit(X, required_bits=40960 - 1424 - 8 - 16, s=s)
    #vlc, hufftab = lbtenc(X, jpeg_quant_size, opthuff=True, log=False, s=s)

    return vlc, hufftab
