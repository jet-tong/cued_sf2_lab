# put any code shared between your encoder and decoder here
from typing import Any
from cued_sf2_lab.jpeg import HuffmanTable

# Here we indicate what format we plan to store our header in. This is just documentation for the
# demonstrator marking it, neither Python nor the competition runner care
HeaderType = HuffmanTable  # using the example jpeg encoder
# HeaderType = None        # when not using a header
# HeaderType = np.ndarray  # when your header is single array
# HeaderType = Any         # if you're using some other header forma

# a constant shared by the encoder and decoder
jpeg_quant_size = 128 # default was 128, not sure 

def my_function():
    pass