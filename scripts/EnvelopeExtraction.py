"""

This script extracts the enveloppe of each 128*nbfiles outputs created by the GammatoneFiltering.py script,
using Hillbert transform and low pass filtering.

"""
from __future__ import division

import glob
import os
import time
from multiprocessing.pool import Pool
from os.path import splitext, join

import matplotlib.pyplot as plt
import numpy
from matplotlib.colors import LogNorm
from scipy.signal import hilbert, butter, lfilter

from .GammatoneFiltering import CENTER_FREQUENCIES
from .FBFileReader import GetF2Frequencies


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
    padding = numpy.zeros(int(2 ** numpy.ceil(numpy.log2(len(signal)))) - len(signal))
    # Append it at the end of the signal
    tohilbert = numpy.hstack((signal, padding))
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
    print(B,A)
    return lfilter(B, A, signal, axis=0)


def ExtractEnvelope(gfbFileName):
    """
    Extracts 128 envelopes from the npy matrix stored in the parameter file
    :param gfbFileName: path to the file to be processed, with the extension .GFB.npy
    """
    print("File:", gfbFileName)
    # Load the matrix
    matrix = numpy.load(gfbFileName)
    # Matrix that will be saved
    envelope=numpy.zeros(matrix.shape)
    # Computing the envelope and filtering it
    for i, signal in enumerate(matrix):
        # print(npyfile,i)
        # Envelope extraction
        analytic_signal = paddedHilbert(signal)
        amplitude_envelope = numpy.abs(analytic_signal)
        if not LP_FILTERING:
            envelope[i] = amplitude_envelope
        else:
            # Low Pass Filter with Butterworth 'CUTOFF' Hz filter
            filtered_envelope_values = lowPassFilter(amplitude_envelope, CUTOFF)
            # Save the envelope to the right output channel
            envelope[i] = filtered_envelope_values

    PlotEnvelopeSpectrogram(envelope, splitext(splitext(gfbFileName)[0])[0])
    print(gfbFileName, "done !")
    return envelope


def SaveEnvelope(matrix, gfbFileName):
    """
    Save the envelope matrix to a file with the extension .ENVx.npy, with x the method used(1,2,...)
    :param matrix: the (128 * nbframes) matrix of envelopes to be saved
    :param gfbFileName: the original filename
    """
    # Envelope file nane is NAME.ENV+METHOD NUMBER.npy (.ENV1,.ENV2...)
    envelopeFilename = splitext(splitext(gfbFileName)[0])[0] + ".ENV" + str(METHOD)
    numpy.save(envelopeFilename, matrix)


def ExtractAndSaveEnvelope(gfbFileName):
    SaveEnvelope(ExtractEnvelope(gfbFileName), gfbFileName)


def ERBScale(f):
    return 24.7*(4.37*f*0.001+1)


def GetNewHeightERB(matrix):
    height=0
    ratios=[]
    base=ERBScale(CENTER_FREQUENCIES[-1])
    for i, line in enumerate(matrix):
        erb=ERBScale(CENTER_FREQUENCIES[i])
        ratio=int(round(erb/base))
        ratios.append(ratio)
        height+=ratio
    return height, ratios


def PlotEnvelopeSpectrogram(matrix, filename):
    # Plotting the image, with logarithmic normalization
    h, ratios=GetNewHeightERB(matrix)
    image=numpy.zeros([h,matrix.shape[1]])
    print(image.shape)
    i=0
    r=0
    print(ratios)
    for line in matrix:
        j=0
        for j in range(ratios[r]):
            image[i+j]=line
        print(i, i + j)

        i+=j+1
        r+=1
    plt.imshow(image, norm=LogNorm(), aspect="auto", extent=[0, len(matrix[0]) / 16000., 100, 7795])

    # plt.pcolormesh(matrix)
    # Get the VTR F2 Formants from the database
    F2Array, sampPeriod = GetF2Frequencies(filename + ".FB")
    t = [i * sampPeriod / 1000000. for i in range(len(F2Array))]
    # TODO: Splitting by / is not platform independent
    file = filename.split('/')
    plt.title("Envelopes of " + os.path.join(file[-2], file[-1]) + ".WAV and F2 formants from VTR database")
    # Plotting the VTR F2 formant over the envelope image
    line, = plt.plot(t, F2Array, "r.")
    plt.legend(("F2 Formant", line))
    plt.show()


def main():
    # # In case you need to print numpy outputs:
    # numpy.set_printoptions(threshold=numpy.inf, suppress=True)
    TotalTime = time.time()

    # # Get all the GFB.npy files under resources/fcnn
    # gfbFiles = glob.glob(join("resources", "f2cnn", "*", "*.GFB.npy"))

    # Test Files
    gfbFiles = glob.glob(join( "testFiles", "*.GFB.npy"))

    if not gfbFiles:
        print("NO GFB.npy FILES FOUND")
        exit(-1)
    print(gfbFiles)

    # Usage of multiprocessing, to reduce computing time
    proc = 4
    multiproc_pool = Pool(processes=proc)
    multiproc_pool.map(ExtractAndSaveEnvelope, gfbFiles)

    print('              Total time:', time.time() - TotalTime)


if __name__ == '__main__':
    main()
