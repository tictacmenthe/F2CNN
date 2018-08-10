"""
    This file gives access to plotting data unrelated to the CNN for this project
"""

import os
import time

import numpy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from gammatone.filters import centre_freqs, make_erb_filters
from scripts.processing.EnvelopeExtraction import ExtractEnvelopeFromMatrix
from scripts.processing.FBFileReader import ExtractFBFile
from scripts.processing.GammatoneFiltering import GetFilteredOutputFromFile


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
    base = ERBScale(CENTER_FREQUENCIES[-1])  # Lowest frequency's bandwith
    for i, line in enumerate(matrix):
        erb = ERBScale(CENTER_FREQUENCIES[i])  # The ERB at this center frequency
        ratio = int(round(erb / base))  # We round up or down the ratio, since pixels are discrete...
        ratios.append(ratio)
        height += ratio
    return height, ratios


def ReshapeEnvelopesForSpectrogram(envelopes, CENTER_FREQUENCIES):
    # Plotting the image, with logarithmic normalization and cell heights corresponding to ERB scale
    h, ratios = GetNewHeightERB(envelopes, CENTER_FREQUENCIES)
    image = numpy.zeros([h, envelopes.shape[1]])
    i = 0
    r = 0
    for line in envelopes:
        j = 0
        for j in range(ratios[r]):
            image[i + j] = line
        i += j + 1
        r += 1
    return image


def PlotEnvelopeSpectrogram(matrix, CENTER_FREQUENCIES):
    """
    Plots a spectrogram-like representation of a matrix, with ERB scale as bandwidths
    :param title: title for the plot
    :param sampPeriod: sampling period of the formants
    :param Formants: 2d array of the formants to plot along the image
    :param matrix: the matrix of outputs from the FilterBank
    :param CENTER_FREQUENCIES: the center frequency of each channel of the matrix
    """
    image = ReshapeEnvelopesForSpectrogram(matrix, CENTER_FREQUENCIES)
    # Plotting the VTR formants over the envelope image

    plt.imshow(image, norm=LogNorm(), aspect="auto", extent=[0, len(matrix[0]) / 16000., 100, 7795])


def PlotEnvelopesAndF2FromFile(filename):
    CENTER_FREQUENCIES = centre_freqs(16000, 128, 100)
    FILTERBANK_COEFFICIENTS = make_erb_filters(16000, CENTER_FREQUENCIES)
    matrix, framerate = GetFilteredOutputFromFile(filename, FILTERBANK_COEFFICIENTS)
    matrix = ExtractEnvelopeFromMatrix(matrix)
    PlotEnvelopeSpectrogram(matrix, CENTER_FREQUENCIES)
    fbPath = os.path.splitext(filename)[0] + '.FB'
    formants, sampPeriod = ExtractFBFile(fbPath)
    if formants is not None:
        Formants = [[], [], [], []]
        formants = formants[:, :4]
        for i in range(4):
            for j in range(len(formants)):
                Formants[i].append(formants[j][i])
        t = [i * sampPeriod / 1000000. for i in range(len(Formants[0]))]
        for i, Formant in enumerate(Formants):
            plt.plot(t, Formant, label='F{} Frequencies (Hz)'.format(i + 1))
        plt.legend()
        plt.text(t[-1] / 2, -700, "File:" + filename)
    title = "'Spectrogram' like representation of envelopes, and formants"
    plt.title(title)
    plt.show()
