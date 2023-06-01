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
    result = minimize(objective1, 10, method='Nelder-Mead')
    final_step = result.x[0]
    Yq = quantise(Y, final_step)
    return Yq


####################################################################################
# Arithmetic encoder and decoder

def encode1(x):

    precision = 32
    one = int(2**precision-1)
    quarter = int(ceil(one/4))
    half = 2*quarter
    threequarters = 3*quarter
    
    #Dictionary of keys: labels and values:frequency
    f_tot = dict([(key, len(list(group))) for key, group in groupby(sorted(x))])
    #total number of characters
    Nin = len(x)
    #probability of our completed dataset
    p_tot = dict([(a,f_tot[a]/Nin) for a in f_tot]) 

    #Initialise probabilities for all variables
    #order = 1
    delta = 1/128
    f = delta * np.ones((128))
    Ltot = 0
    Lav = []
    
    y = [] # initialise output list
    lo,hi = 0,one # initialise lo and hi to be [0,1.0)
    straddle = 0 # initialise the straddle counter to 0

    
    for k in range(len(x)): # for every symbol

        # arithmetic coding is slower than vl_encode, so we display a "progress bar"
        # to let the user know that we are processing the file and haven't crashed...
        if k % 100 == 0:
            so.write('Arithmetic encoded %d%%    \r' % int(floor(k/len(x)*100)))
            so.flush()

        # 1) calculate the interval range to be the difference between hi and lo and 
        # add 1 to the difference. The added 1 is necessary to avoid rounding issues
        # lohi_range = ....
        lohi_range = hi-lo+1
        # 2) narrow the interval end-points [lo,hi) to the new range [f,f+p]
        # within the old interval [lo,hi], being careful to round 'innwards' so
        # the code remains prefix-free (you want to use the functions ceil and
        # floor). This will require two instructions. Note that we start computing
        # the new 'lo', then compute the new 'hi' using the scaled probability as
        # the offset from the new 'lo' to the new 'hi'
        # ...
        # ...
        key = ord(x[k])
        p = [x/sum(f) for x in f]
        cdf = [0]
        for _ in p:
            su = cdf[-1] + _
            cdf.append(su)
        cdf.pop()
        
        lo = lo+ceil(lohi_range*cdf[key])
        hi = lo + floor(lohi_range*p[key])
        f[key] += 1
        


        if (lo == hi):
            raise NameError('Zero interval!')

        # Now we need to re-scale the interval if its end-points have bits in common,
        # and output the corresponding bits where appropriate. We will do this with an
        # infinite loop, that will break when none of the conditions for output / straddle
        # are fulfilled
        while True:
            if hi < half: # if lo < hi < 1/2
                # stretch the interval by 2 and output a 0 followed by 'straddle' ones (if any)
                # and zero the straddle after that. In fact, HOLD OFF on doing the stretching:
                # we will do the stretching at the end of the if statement
                # ...  append a zero to the output list y
                # ...  extend by a sequence of 'straddle' ones
                # ...  zero the straddle counter
                y.append(0)
                y.extend([1]*straddle)
                straddle = 0
            elif lo >= half: # if hi > lo >= 1/2
                # stretch the interval by 2 and substract 1, and output a 1 followed by 'straddle'
                # zeros (if any) and zero straddle after that. Again, HOLD OFF on doing the stretching
                # as this will be done after the if statement, but note that 2*interval - 1 is equivalent
                # to 2*(interval - 1/2), so for now just substract 1/2 from the interval upper and lower
                # bound (and don't forget that when we say "1/2" we mean the integer "half" we defined
                # above: this is an integer arithmetic implementation!
                # ...  append a 1 to the output list y
                # ...  extend 'straddle' zeros
                # ...  reset the straddle counter
                # ...
                # ...  substract half from lo and hi
                y.append(1)
                y.extend([0]*straddle)
                straddle = 0
                hi -= half
                lo -= half
            elif lo >= quarter and hi < threequarters: # if 1/4 < lo < hi < 3/4
                # we can increment the straddle counter and stretch the interval around
                # the half way point. This can be impemented again as 2*(interval - 1/4),
                # and as we will stretch by 2 after the if statement all that needs doing
                # for now is to subtract 1/4 from the upper and lower bound
                # ...  increment straddle
                # ...
                # ...  subtract 'quarter' from lo and hi
                straddle += 1
                lo -= quarter
                hi -= quarter
            else:
                break # we break the infinite loop if the interval has reached an un-stretchable state
            # now we can stretch the interval (for all 3 conditions above) by multiplying by 2
            lo *= 2
            hi = 2*hi+1
            # ...  multiply lo by 2
            # ...  multiply hi by 2 and add 1 (I DON'T KNOW WHY +1 IS NECESSARY BUT IT IS. THIS IS MAGIC.
            #      A BOX OF CHOCOLATES FOR ANYONE WHO GIVES ME A WELL ARGUED REASON FOR THIS... It seems
            #      to solve a minor precision problem.)

    # termination bits
    # after processing all input symbols, flush any bits still in the 'straddle' pipeline
    straddle += 1 # adding 1 to straddle for "good measure" (ensures prefix-freeness)
    if lo < quarter: # the position of lo determines the dyadic interval that fits
        # ...  output a zero followed by "straddle" ones
        # ...
        y.append(0)
        y.extend([1]*straddle)
    else:
        # ...  output a 1 followed by "straddle" zeros
        y.append(1)
        y.extend([0]*straddle)


    return(y)

def decode1(y,n):
    precision = 32
    one = int(2**precision - 1)
    quarter = int(ceil(one/4))
    half = 2*quarter
    threequarters = 3*quarter

    alphabetsize = 128
    alphabet = []
    for i in range(alphabetsize):
        alphabet.append(chr(i))
    
    

    y.extend(precision*[0]) # dummy zeros to prevent index out of bound errors
    x = n*[0] # initialise all zeros 

    # initialise by taking first 'precision' bits from y and converting to a number
    value = int(''.join(str(a) for a in y[0:precision]), 2) 
    y_position = precision # position where currently reading y
    lo,hi = 0,one

    x_position = 0
    delta = 1
    f_tot = delta*np.ones((128))
    while 1:
        
        p = [x/sum(f_tot) for x in f_tot]

        cdf = [0]
        for a in p:
            cdf.append(cdf[-1]+a)
        cdf.pop()

        #if x_position % 100 == 0:
        #    so.write('Arithmetic decoded %d%%    \r' % int(floor(x_position/n*100)))
        #    so.flush()

        lohi_range = hi - lo + 1
        a = bisect(cdf, (value-lo)/lohi_range) - 1
        x[x_position] = alphabet[a]
        f_tot[ord(x[x_position])] += 1

        
        lo = lo + int(ceil(cdf[a]*lohi_range))
        hi = lo + int(floor(p[a]*lohi_range))
        if (lo == hi):
            raise NameError('Zero interval!')

        while True:
            if hi < half:
                # do nothing
                pass
            elif lo >= half:
                lo = lo - half
                hi = hi - half
                value = value - half
            elif lo >= quarter and hi < threequarters:
                lo = lo - quarter
                hi = hi - quarter
                value = value - quarter
            else:
                break
            lo = 2*lo
            hi = 2*hi + 1
            value = 2*value + y[y_position]
            y_position += 1
            if y_position == len(y):
                break
        
        x_position += 1    
        if x_position == n or y_position == len(y):
            break
        
    return(x)
    

    
