"""

This script extracts the enveloppe of each 128*nbfiles outputs created by the GammatoneFiltering.py script,
using Hillbert transform and low pass filtering.

"""
from __future__ import division

import glob
import time
from itertools import repeat
from multiprocessing import cpu_count, Value
from multiprocessing.pool import Pool
from os.path import splitext, join, split

import numpy
from scipy.signal import hilbert, lfilter, butter


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
    B, A = butter(1, freq / (16000 / 2), 'low')
    return lfilter(B, A, signal, axis=0)


def ExtractEnvelopeFromMatrix(matrix, LPF=False, CUTOFF=100):
    # Matrix that will be saved
    envelopes = numpy.zeros(matrix.shape)
    # Computing the envelope and filtering it
    for i, signal in enumerate(matrix):
        # Envelope extraction
        analytic_signal = paddedHilbert(signal)
        amplitude_envelope = numpy.abs(analytic_signal)
        # If not Low Pass Filtering
        if not LPF:
            envelopes[i] = amplitude_envelope
        else:
            # Low Pass Filter with Butterworth 'CUTOFF' Hz filter
            filtered_envelope_values = lowPassFilter(amplitude_envelope, CUTOFF)
            # Save the envelope to the right output channel
            envelopes[i] = filtered_envelope_values
    return envelopes


def ExtractEnvelope(gfbFileName, nbf, LPF=False, CUTOFF=100):
    """
    Extracts 128 envelopes from the npy matrix stored in the parameter file
    :param LPF: boolean for whether or not using low pass filtering
    :param CUTOFF: cutoff frequency of the LPF
    :param nbf: number of files, for printing
    :param gfbFileName: path to the file to be processed, with the extension .GFB.npy
    """
    print("File:\t\t{}".format(gfbFileName))
    # Load the matrix
    matrix = numpy.load(gfbFileName)
    envelopes = ExtractEnvelopeFromMatrix(matrix, LPF, CUTOFF)
    global counter
    with counter.get_lock():
        counter.value += 1
        print("\t\t{:<50} done ! {}/{} Files.".format(gfbFileName, counter.value, nbf))
    return envelopes


def SaveEnvelope(matrix, gfbFileName):
    """
    Save the envelope matrix to a file with the extension .ENVx.npy, with x the method used(1,2,...)
    :param matrix: the (128 * nbframes) matrix of envelopes to be saved
    :param gfbFileName: the original filename
    """
    METHOD = 1
    # Envelope file nane is NAME.ENV+METHOD NUMBER.npy (.ENV1,.ENV2...)
    envelopeFilename = splitext(splitext(gfbFileName)[0])[0] + ".ENV" + str(METHOD)
    numpy.save(envelopeFilename, matrix)


def ExtractAndSaveEnvelope(gfbFileName, nbf, LPF=False, CUTOFF=100):
    """
    :param gfbFileName: path to the .GFB.npy file to use
    :param nbf: total number of files
    :param LPF: boolean for whether or not using low pass filtering
    :param CUTOFF: cutoff frequency of the LPF
    """
    SaveEnvelope(ExtractEnvelope(gfbFileName, nbf, LPF, CUTOFF), gfbFileName)


def InitProcesses(cn):
    global counter
    counter = cn


def ExtractAllEnvelopes(LPF=False, CUTOFF=100):
    # # In case you need to print numpy outputs:
    # numpy.set_printoptions(threshold=numpy.inf, suppress=True)
    TotalTime = time.time()

    # Get all the GFB.npy files under resources/fcnn
    gfbFiles = glob.glob(join("resources", "f2cnn", "*", "*.GFB.npy"))
    print("\n###############################\nExtracting Envelopes from files in '{}'.".format(split(gfbFiles[0])[0]))
    if LPF:
        print("Using Low Pass Filtering with a cutoff at {}Hz".format(CUTOFF))
    else:
        print("Not using Low Pass Filtering")

    if not gfbFiles:
        print("ERROR: NO .GFB.npy FILES FOUND")
        exit(-1)

    print(len(gfbFiles), ".GFB.npy files found")

    # Usage of multiprocessing, to reduce computing time
    proc = cpu_count()
    counter = Value('i', 0)
    multiproc_pool = Pool(processes=proc, initializer=InitProcesses, initargs=(counter,))
    arguments = zip(gfbFiles, repeat(len(gfbFiles)), repeat(LPF), repeat(CUTOFF))           # Pack all the arguments
    multiproc_pool.starmap(ExtractAndSaveEnvelope, arguments)

    print("Extracted Envelopes from all files.")
    print('              Total time:', time.time() - TotalTime)
    print('')
