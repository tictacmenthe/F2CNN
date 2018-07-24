from os.path import join

import matplotlib.pyplot as plt
import numpy
from matplotlib.colors import LogNorm

from .OrganiseFiles import completeSplit
from .FBFileReader import GetF2Frequencies


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


def PlotEnvelopeSpectrogramWithF2(matrix, CENTER_FREQUENCIES, Formants, sampPeriod=10000, title=""):
    """
    Plots a spectrogram-like representation of a matrix, with ERB scale as bandwidths, and filename gives the
    :param title: title for the plot
    :param sampPeriod: sampling period of the formants
    :param Formants: 2d array of the formants to plot along the image
    :param matrix: the matrix of outputs from the FilterBank
    :param CENTER_FREQUENCIES: the center frequency of each channel of the matrix
    """
    # Plotting the image, with logarithmic normalization and cell heights corresponding to ERB scale
    h, ratios = GetNewHeightERB(matrix)
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

    # plt.pcolormesh(matrix)
    # Get the VTR F2 Formants from the database
    # F2Array, sampPeriod = GetF2Frequencies(filename + ".FB")

    t = [i * sampPeriod / 1000000. for i in range(len(Formants[0]))]
    # splitted=completeSplit(filename)
    # print(splitted)
    # plt.title("Envelopes of " + join(*splitted) + ".WAV and F2 formants from VTR database")
    # Plotting the VTR F2 formant over the envelope image
    plt.title(title)
    lines=[]
    for i, formant in enumerate(Formants):
        line, = plt.plot(t, formant)
        lines.append(line)
    plt.legend(lines, ("F{} Frequencies (Hz)".format(i+1) for i in range(len(lines))))
    plt.show()
