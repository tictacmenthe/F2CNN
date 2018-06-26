"""

This script extracts the enveloppe of each 128*nbfiles outputs created by the GammatoneFiltering.py script,
using Hillbert transform and low pass filtering.

"""
from __future__ import division

import glob
import time
from os.path import splitext

from multiprocessing.pool import Pool
import numpy as np
from scipy.signal import hilbert, butter, lfilter

SAMPLING_RATE = 16000
CUTOFF = 100
METHOD = 1


def paddedHilbert(signal):
    """
    Computes the analytic signal of 'signal' with a fast hilbert transform
    Considering FFTs are very slow when the length of the signal is not a power of 2, this pads with zeroes the signal
    for a very fast hilber transform, then cuts it back to the correct length
    :param signal: the signal to use for analytic signal computation
    :return: the analytic signal
    """
    padding = np.zeros(int(2 ** np.ceil(np.log2(len(signal)))) - len(signal))
    tohilbert = np.hstack((signal, padding))

    result = hilbert(tohilbert)

    result = result[0:len(signal)]

    return result


def lowPassFilter(signal, freq):
    """
    Applies a butterworth low pass filter to the signal
    :param signal: the signal that will be filtered
    :param freq: the cutoff frequency
    :return: the filtered signal
    """
    # The A et B parameter arrays of the filter
    B, A = butter(1, freq / (SAMPLING_RATE / 2), 'low')
    return lfilter(B, A, signal, axis=0)


def EnvelopeExtraction(npyfile):
    """
    The function that is used by the multiprocessing pool for faster computation of the envelopes
    Saves the filtered enveloped that was extracted to the ORIGINALNAME.ENVx.npy format,
    with x being the method used(1,2,3...)
    :param npyfile: path to the file to be processed
    """
    # Load the matrix
    matrix = np.load(npyfile)

    # Matrix that will be saved
    outputMatrix = np.zeros(matrix.shape)

    # Computing the envelope and filtering it
    duration = time.time()
    for i, signal in enumerate(matrix):
        # print(npyfile,i)
        # Envelope extraction
        analytic_signal = paddedHilbert(signal)
        amplitude_envelope = np.abs(analytic_signal)
        # Low Pass Filter with Butterworth 'CUTOFF' Hz filter
        filtered_envelope = lowPassFilter(amplitude_envelope, CUTOFF)
        # Save the envelope to the right output channel
        outputMatrix[i] = filtered_envelope

    duration = time.time() - duration

    print("ENDED", npyfile)
    print(len(matrix[0]))
    print("        Hilbert Duration:", duration)

    # Saving the matrix
    duration = time.time()
    # Envelope file nane is NAME.ENV+METHOD NUMBER.npy (.ENV1,.ENV2...)
    envelopeFilename = splitext(splitext(npyfile)[0])[0] + ".ENV" + str(METHOD)
    np.save(envelopeFilename, outputMatrix)
    duration = time.time() - duration
    print("         Saving Duration:", duration)


def main():
    # # In case you need to print numpy outputs:
    # numpy.set_printoptions(threshold=numpy.inf, suppress=True)
    TotalTime = time.time()

    # Get all the WAV files under ../src
    gfbFiles = glob.glob('../src/f2cnn/*/*.GFB.npy')

    # # Test WavFiles
    # gfbFiles = ["../src/f2cnn/TRAIN/DR6.MEAL0.SX197.GFB.npy"]

    # Usage of multiprocessing, to reduce computing time
    proc = 4
    multiproc_pool = Pool(processes=proc)
    multiproc_pool.map(EnvelopeExtraction, gfbFiles)

    print('              Total time:', time.time() - TotalTime)


if __name__ == '__main__':
    main()
