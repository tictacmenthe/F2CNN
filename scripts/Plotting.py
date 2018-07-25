import os

import matplotlib.pyplot as plt
import numpy
from matplotlib.colors import LogNorm

from gammatone.filters import centre_freqs
from .EnvelopeExtraction import ExtractEnvelopeFromMatrix
from .FBFileReader import ExtractFBFile
from .GammatoneFiltering import GetFilteredOutputFromFile


def ERBScale(f):
    """
    Computes the Equivalent Rectangular Bandwith at center frequency f, using Moore and Glasberg's linear approximation
    :param f:   The center frequency to be considered, in Hz
    :return:    The resulting bandwith
    """
    return 24.7 * (4.37 * f * 0.001 + 1)


def GetNewHeightERB(matrix, CENTER_FREQUENCIES):
    """
    Compute the new height of the image if every line was multiplied by the the ratio to the bw of the lowest frequency
    :param matrix: the matrix of outputs from the FilterBank
    :param CENTER_FREQUENCIES: the center frequency of each channel of the matrix
    :return: the new height of the image, and each value of the ratios
    """
    height = 0
    ratios = []
    base = ERBScale(CENTER_FREQUENCIES[-1])     # Lowest frequency's bandwith
    for i, line in enumerate(matrix):
        erb = ERBScale(CENTER_FREQUENCIES[i])   # The ERB at this center frequency
        ratio = int(round(erb / base))          # We round up or down the ratio, since pixels are discrete...
        ratios.append(ratio)
        height += ratio
    return height, ratios


def PlotEnvelopeSpectrogram(matrix, CENTER_FREQUENCIES, Formants, sampPeriod=10000, title=""):
    """
    Plots a spectrogram-like representation of a matrix, with ERB scale as bandwidths, and filename gives the
    :param title: title for the plot
    :param sampPeriod: sampling period of the formants
    :param Formants: 2d array of the formants to plot along the image
    :param matrix: the matrix of outputs from the FilterBank
    :param CENTER_FREQUENCIES: the center frequency of each channel of the matrix
    """
    # Plotting the image, with logarithmic normalization and cell heights corresponding to ERB scale
    h, ratios = GetNewHeightERB(matrix, CENTER_FREQUENCIES)
    image = numpy.zeros([h, matrix.shape[1]])
    i = 0
    r = 0
    for line in matrix:
        j = 0
        for j in range(ratios[r]):
            image[i + j] = line
        i += j + 1
        r += 1
    plt.imshow(image, norm=LogNorm(), aspect="auto", extent=[0, len(matrix[0]) / 16000., 100, 7795])

    # Plotting the VTR formants over the envelope image
    formants=[[],[],[],[]]
    for i in range(4):
        for j in range(len(Formants)):
            formants[i].append(Formants[j][i])
    t = [i * sampPeriod / 1000000. for i in range(len(formants[0]))]

    plt.title(title)
    for i, formant in enumerate(formants):
        plt.plot(t, formant, label='F{} Frequencies (Hz)'.format(i))
    plt.legend()
    plt.show()


def PlotEnvelopesAndF2FromFile(filename):
    matrix = GetFilteredOutputFromFile(filename)
    matrix=ExtractEnvelopeFromMatrix(matrix)
    CENTER_FREQUENCIES = centre_freqs(16000, 128, 100)
    formants, sampPeriod = ExtractFBFile(os.path.splitext(filename)[0]+'.FB')
    formants=formants[:,:4]
    title="'Spectrogram' like representation of envelopes, and formants"
    PlotEnvelopeSpectrogram(matrix,CENTER_FREQUENCIES,formants, sampPeriod, title)
