#!/usr/bin/env python

"""
Copyright (C) 2003-2014, Michele Cappellari
E-mail: michele.cappellari_at_physics.ox.ac.uk

    V1.0.0: Michele Cappellari, Vicenza, 13 February 2003
    V1.0.1: Use astro library routines to read and write files.
        MC, Leiden, 24 July 2003
    V2.0.0: Translated from IDL into Python. MC, London, 19 March 2014
    V2.0.1: Support both Python 2.6/2.7 and Python 3.x. MC, Oxford, 25 May 2014
    V2.0.2: Make files paths relative to this file, to run the example from
        any directory. MC, Oxford, 23 January 2017
    V2.0.3: Changed imports for vorbin as package. 
        Make file paths relative to the vorbin package to be able to run the
        example unchanged from any directory. MC, Oxford, 17 April 2018    
    V2.0.4: Dropped legacy Python 2.7 support. MC, Oxford, 10 May 2018

"""

from os import path
import numpy as np
import matplotlib.pyplot as plt

import vorbin
from vorbin.voronoi_2d_binning import voronoi_2d_binning

#-----------------------------------------------------------------------------

def voronoi_binning_example():
    """
    Usage example for the procedure VORONOI_2D_BINNING.

    It is assumed below that the file voronoi_2d_binning_example.txt
    resides in the current directory. Here columns 1-4 of the text file
    contain respectively the x, y coordinates of each SAURON lens
    and the corresponding Signal and Noise.

    """
    file_dir = path.dirname(path.realpath(vorbin.__file__))  # path of vorbin
    y, x, signal, noise = np.loadtxt("GoodPixels.txt").T
    targetSN = 20.0

    # Perform the actual computation. The vectors
    # (binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale)
    # are all generated in *output*
    #
    binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = voronoi_2d_binning(
        x, y, signal, noise, targetSN, plot=1, quiet=0)

    # Save to a text file the initial coordinates of each pixel together
    # with the corresponding bin number computed by this procedure.
    # binNum uniquely specifies the bins and for this reason it is the only
    # number required for any subsequent calculation on the bins.
    #
    np.savetxt('voronoi_2d_binning_example_output.txt', np.column_stack([x, y, binNum]),
               fmt=b'%10.6f %10.6f %8i')

#-----------------------------------------------------------------------------

if __name__ == '__main__':

    voronoi_binning_example()
    plt.tight_layout()
    plt.plot()
    plt.show()
