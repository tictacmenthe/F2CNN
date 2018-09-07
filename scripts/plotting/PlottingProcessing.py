"""
    This file gives access to plotting data unrelated to the CNN for this project
"""

import os
from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy
from matplotlib.colors import LogNorm

from gammatone.filters import centre_freqs, make_erb_filters
from scripts.processing.EnvelopeExtraction import ExtractEnvelopeFromMatrix
from scripts.processing.FBFileReader import ExtractFBFile
from scripts.processing.GammatoneFiltering import GetFilteredOutputFromFile, GetArrayFromWAV


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


def ReshapeEnvelopesForSpectrogram(envelopes, CENTER_FREQUENCIES, start=0, end=None):
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
    if end is not None:
        return image[:, start:end]
    else:
        return image[:, start:]


def PlotEnvelopeSpectrogram(matrix, CENTER_FREQUENCIES, axis=plt, LOW_FREQ=100, FRAMERATE=16000, start=0, end=None):
    """
    Plots a spectrogram-like representation of a matrix, with ERB scale as bandwidths
    :param matrix: the matrix of outputs from the FilterBank
    :param CENTER_FREQUENCIES: the center frequency of each channel of the matrix
    :param LOW_FREQ: Plotting frequency range's lower limit
    :param FRAMERATE: Framerate used for the .WAV file
    :param start: starting point of the plot, in seconds
    :param end: ending point of the plot
    :return end: the new end point of the matrix, in case it was None
    """
    image = ReshapeEnvelopesForSpectrogram(matrix, CENTER_FREQUENCIES, start, end)
    # Plotting the VTR formants over the envelope image
    axis.imshow(image, norm=LogNorm(), aspect="auto",
               extent=[start, len(image[0]) / FRAMERATE, LOW_FREQ, int(FRAMERATE / 2)])
    return len(image)


def PlotEnvelopesAndFormantsFromFile(filename, start=0, end=None, formantToPlot=5):
    """
    Plots a spectrogramlike representation of the gammatone filterbank output on a wav file,
    including formants from VTR database if available
    :type filename: path to a .wav file, if there are .FB files for formant data,
                    they should have the same name and end in .FB
    :param start: starting point of the plot, in seconds
    :param end: ending point of the plot
    :param formantToPlot: specific formant to plot
    """
    # #### READING CONFIG FILE
    config = ConfigParser()
    config.read('configF2CNN.conf')
    LOW_FREQ = config.getint('FILTERBANK', 'LOW_FREQ')
    sampPeriod = config.getint('CNN', 'SAMPLING_PERIOD')

    framerate, _ = GetArrayFromWAV(filename)
    ustos = 1.0 / 1000000
    CENTER_FREQUENCIES = centre_freqs(framerate, 128, 100)
    FILTERBANK_COEFFICIENTS = make_erb_filters(framerate, CENTER_FREQUENCIES)
    matrix, framerate = GetFilteredOutputFromFile(filename, FILTERBANK_COEFFICIENTS)
    matrix = ExtractEnvelopeFromMatrix(matrix)

    # Plot the gtgram but do not show it, changes end to the size(if it was None)
    end = PlotEnvelopeSpectrogram(matrix, CENTER_FREQUENCIES=CENTER_FREQUENCIES, LOW_FREQ=LOW_FREQ, start=start,
                                  end=end)

    fbPath = os.path.splitext(filename)[0] + '.FB'
    formants, _ = ExtractFBFile(fbPath)
    print(len(formants))
    print(sampPeriod)

    # Plot the formants, if available
    if formants is not None:
        Formants = [[], [], [], []]
        formants = formants[:, :4]  # The formants are the first 4 columns of the .FB file(which is a binary file)
        for i in range(4):
            for j in range(len(formants)):
                Formants[i].append(formants[j][i])
        # Time range for the plot, in seconds
        t = [i * sampPeriod * ustos for i in range(len(Formants[0]))]

        if 0 < formantToPlot < 5:
            Formants = Formants[formantToPlot-1]
            plt.plot(t,Formants, label='F{} Frequencies (Hz)'.format(formantToPlot))
        else:
            for i, Formant in enumerate(Formants):
                plt.plot(t, Formant, label='F{} Frequencies (Hz)'.format(i + 1))
        plt.legend()
        plt.text(t[-1] / 2, -700, "File:" + filename)
    title = "'Spectrogram' like representation of envelopes, and formants"
    plt.title(title)
    plt.show()
