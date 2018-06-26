"""

This script extracts the enveloppe of each 128*nbfiles outputs created by the GammatoneFiltering.py script,
using Hillbert transform and low pass filtering.

"""
import glob
import time

from scipy.signal import hilbert
import numpy
import matplotlib.pyplot as plt






def main():
    # # In case you need to print numpy outputs:
    # numpy.set_printoptions(threshold=numpy.inf, suppress=True)
    TotalTime = time.time()

    # Get all the WAV files under ../src
    gfbFiles = glob.glob('../src/f2cnn/*/*.npy')

    # # Test WavFiles
    # wavFiles = ['testSmall.WAV', 'testBig.WAV']

    averageSavingDuration = 0
    averageFilterDuration = 0
    for i, w in enumerate(gfbFiles):


    print('\nAverage Filtering duration:', float(averageSavingDuration) / len(gfbFiles))
    print('   Average Saving duration:', float(averageFilterDuration) / len(gfbFiles))
    print('                Total time:', time.time() - TotalTime)


if __name__ == '__main__':
    main()
