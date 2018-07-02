"""

This script extracts the enveloppe of each 128*nbfiles outputs created by the GammatoneFiltering.py script,
using Hillbert transform and low pass filtering.

"""
from __future__ import division

import glob
import os
import time
from os.path import splitext, join

from multiprocessing.pool import Pool
import numpy as np
from scipy.signal import hilbert, butter, lfilter
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scripts.FBFileReader import ExtractFBFile

SAMPLING_RATE = 16000
CUTOFF = 100
METHOD = 1
LP_FILTERING = False


def paddedHilbert(signal):
    """
    Computes the analytic signal of 'signal' with a fast hilbert transform
    FFTs are very slow when the length of the signal is not a power of 2 or is far from it,
    this pads with zeroes the signal for a very fast hilber transform, then cuts it back to the correct length
    :param signal: the signal to use for analytic signal computation
    :return: the analytic signal
    """
    # Array of 0 for padding until the next power of 2
    padding = np.zeros(int(2 ** np.ceil(np.log2(len(signal)))) - len(signal))
    # Append it at the end of the signal
    tohilbert = np.hstack((signal, padding))
    # Hilbert transform with the padded signal
    result = hilbert(tohilbert)
    # Remove excess values
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


def ExtractEnvelope(gfbFileName):
    """
    Extracts 128 envelopes from the npy matrix stored in the parameter file
    :param gfbFileName: path to the file to be processed, with the extension .GFB.npy
    """
    print("File:", gfbFileName)
    # Load the matrix
    matrix = np.load(gfbFileName)

    # Matrix that will be saved
    if LP_FILTERING:
        filteredEnvelope = np.zeros(matrix.shape)
    else:
        unfilteredEnvelope = np.zeros(matrix.shape)
    # Computing the envelope and filtering it
    for i, signal in enumerate(matrix):
        # print(npyfile,i)
        # Envelope extraction
        analytic_signal = paddedHilbert(signal)
        amplitude_envelope = np.abs(analytic_signal)
        if not LP_FILTERING: unfilteredEnvelope[i] = amplitude_envelope
        if LP_FILTERING:
            # Low Pass Filter with Butterworth 'CUTOFF' Hz filter
            filtered_envelope = lowPassFilter(amplitude_envelope, CUTOFF)
            # Save the envelope to the right output channel
            filteredEnvelope[i] = filtered_envelope
    PlotEnvelopeSpectrogram(unfilteredEnvelope,splitext(splitext(gfbFileName)[0])[0])
    print(gfbFileName, "done !")
    if LP_FILTERING:
        return filteredEnvelope
    else:
        return unfilteredEnvelope


def SaveEnvelope(matrix, gfbFileName):
    """
    Save the envelope matrix to a file with the extension .ENVx.npy, with x the method used(1,2,...)
    :param matrix: the (128 * nbframes) matrix of envelopes to be saved
    :param gfbFileName: the original filename
    """
    # Envelope file nane is NAME.ENV+METHOD NUMBER.npy (.ENV1,.ENV2...)
    envelopeFilename = splitext(splitext(gfbFileName)[0])[0] + ".ENV" + str(METHOD)
    np.save(envelopeFilename, matrix)


def ExtractAndSaveEnvelope(gfbFileName):
    SaveEnvelope(ExtractEnvelope(gfbFileName), gfbFileName)


def PlotEnvelopeSpectrogram(matrix,filename):
    # Plotting the image, with logarithmic normalization
    plt.imshow(matrix, aspect="auto", extent=[0, len(matrix[0]) / 16000., 100, 7795])
    # Get the VTR F2 Formants from the database
    F2Array, sampPeriod = ExtractFBFile(filename+".FB")
    t = [i * sampPeriod / 1000000. for i in range(len(F2Array))]

    # TODO: Splitting by / is not platform independent
    file=filename.split('/')
    plt.title("Envelopes of "+os.path.join(file[-2],file[-1])+".WAV and F2 formants from VTR database")
    # Plotting the VTR F2 formant over the envelope image
    line, = plt.plot(t, F2Array, "r.")
    plt.legend(("F2 Formant", line))
    plt.show()


def main():
    # # In case you need to print numpy outputs:
    # numpy.set_printoptions(threshold=numpy.inf, suppress=True)
    TotalTime = time.time()

    # # Get all the WAV files under ../src
    # gfbFiles = glob.glob(join("..","src","f2cnn","*","*.GFB.npy"))

    # Test WavFiles
    gfbFiles = ["../src/f2cnn/TEST/DR1.FELC0.SI1386.GFB.npy"]
    # glob.glob(join("..", "testFiles", "smallest", "*.GFB.npy"))[0],
    # glob.glob(join("..", "testFiles", "biggest", "*.GFB.npy"))[0]]
    print(gfbFiles)

    # Usage of multiprocessing, to reduce computing time
    proc = 4
    multiproc_pool = Pool(processes=proc)
    multiproc_pool.map(ExtractAndSaveEnvelope, gfbFiles)

    print('              Total time:', time.time() - TotalTime)


if __name__ == '__main__':
    main()
