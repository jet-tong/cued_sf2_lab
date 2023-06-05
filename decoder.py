""" This file contains the `decode` function. Feel free to split it into smaller functions """
import numpy as np
from cued_sf2_lab.jpeg import jpegdec
from common import my_function, HeaderType, jpeg_quant_size
from jteg import *


def decode(vlc: np.ndarray, header: HeaderType,jpeg_quant_size: float = 48.115, s: float = 1.6099190603156157) -> np.ndarray:
    """
    Parameters:
        X: the input grayscale image
    
    Outputs:
        vlc: the variable-length codes
        header: any additional parameters to be saved alongside the image
    """
    # replace this with your chosen decoding scheme
    #return jpegdec(vlc, jpeg_quant_size, hufftab=header, log=False)
    return lbtdec(vlc, jpeg_quant_size, hufftab=header, log=False, s=s)
