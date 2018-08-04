"""
    This file gives access to plotting data for this project
"""

import os
import time

import matplotlib.pyplot as plt
import numpy
from matplotlib.colors import LogNorm

from gammatone.filters import centre_freqs
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


def PlotEnvelopeSpectrogram(matrix, CENTER_FREQUENCIES, Formants=None, sampPeriod=10000, title=""):
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
    formants = [[], [], [], []]
    if Formants is not None:
        for i in range(4):
            for j in range(len(Formants)):
                formants[i].append(Formants[j][i])
        t = [i * sampPeriod / 1000000. for i in range(len(formants[0]))]

        plt.title(title)
        for i, formant in enumerate(formants):
            plt.plot(t, formant, label='F{} Frequencies (Hz)'.format(i + 1))
        plt.legend()
    plt.imshow(image, norm=LogNorm(), aspect="auto", extent=[0, len(matrix[0]) / 16000., 100, 7795])
    plt.show()


def PlotEnvelopesAndF2FromFile(filename):
    t = time.time()
    CENTER_FREQUENCIES = centre_freqs(16000, 128, 100)
    matrix = GetFilteredOutputFromFile(filename)
    matrix = ExtractEnvelopeFromMatrix(matrix)
    fbPath = os.path.splitext(filename)[0] + '.FB'
    formants, sampPeriod = ExtractFBFile(fbPath)
    if formants is not None:
        formants = formants[:, :4]
    print('compute time:', time.time() - t)
    title = "'Spectrogram' like representation of envelopes, and formants"
    PlotEnvelopeSpectrogram(matrix, CENTER_FREQUENCIES, formants, sampPeriod, title)


def PlotEnvelopesAndCNNResultsWithPhonemes(envelopes, rising, falling, CENTER_FREQUENCIES, phonemes, Formants=None, sampPeriod=0, title=""):
    image = ReshapeEnvelopesForSpectrogram(envelopes, CENTER_FREQUENCIES)
    formant = []
    fig = plt.figure(figsize=(25, 13))
    aximg = fig.add_subplot(211)
    aximg.imshow(image, norm=LogNorm(), aspect="auto", extent=[0, len(envelopes[0]) / 16000., 100, 7795])
    aximg.autoscale(False)
    if Formants is not None:
        for j in range(len(Formants)):
            formant.append(Formants[j][1])
        xformant = [i * sampPeriod / 1000000. for i in range(len(formant))]
        aximg.plot(xformant, formant, 'k-', label='F{} Frequencies (Hz)'.format(2))
    xres = numpy.linspace(0.055, len(image[0])/16000-0.055, len(rising))
    aximg.plot(xres, rising, 'r|', label='Rising')
    aximg.plot(xres, falling, 'b|', label='Falling')
    aximg.set_xlabel("Time(s)")
    aximg.set_ylabel("Frequency(Hz)")
    axproba = fig.add_subplot(212)
    axproba.axis([0,len(image[0])/16000, 0, 1])
    # Plotting the phonemes
    for phoneme, start, end in phonemes:
        axproba.axvline(end/16000)
        axproba.text((end+start)/32000, 0.95, phoneme, fontsize=14, horizontalalignment='center', verticalalignment='center', weight='bold')


    # GetLabelsFromFile('resources/f2cnn/TEST/DR1.FELC0.SX216.WAV')
    plt.legend()
    plt.title(title)
    # plt.show()
    plt.savefig(os.path.join("graphs", "FallingOrRising", "FallingOrRising" + os.path.split(title)[1] + ".png"), dpi=100)
    plt.close(fig)
