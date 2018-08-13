"""
    This file gives access to plotting CNN result data for this project
"""

import os
from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy
from matplotlib.colors import LogNorm
from scipy.stats import pearsonr

from scripts.plotting.PlottingProcessing import ReshapeEnvelopesForSpectrogram


def PlotEnvelopesAndCNNResultsWithPhonemes(envelopes, scores, CENTER_FREQUENCIES, phonemes, Formants=None, title=""):
    image = ReshapeEnvelopesForSpectrogram(envelopes, CENTER_FREQUENCIES)
    formant = []
    # fig = plt.figure()
    fig = plt.figure(figsize=(35,18))
    aximg = fig.add_subplot(211)
    axproba = fig.add_subplot(212)
    axproba.axis([0, len(image[0]) / 16000, -1.6, 1.6])

    aximg.imshow(image, norm=LogNorm(), aspect="auto", extent=[0, len(envelopes[0]) / 16000., 100, 7795])
    aximg.autoscale(False)
    if Formants is not None:
        for j in range(len(Formants)):
            formant.append(Formants[j][1])

        # #### READING CONFIG FILE
        config = ConfigParser()
        config.read('F2CNN.conf')
        radius = config.getint('CNN', 'RADIUS')
        sampPeriod = config.getint('CNN', 'sampperiod') / 1000000
        pvalues = []

        # Discretization of the values for each entry required
        slopes = []
        xformant = [i * sampPeriod for i in range(len(formant))]
        aximg.plot(xformant, formant, 'k-', label='F{} Frequencies (Hz)'.format(2))
        for centerDot in range(radius, len(formant) - radius, 1):
            currentDots = numpy.array([formant[centerDot + (k - 5)] for k in range(11)])
            x = numpy.array([xformant[centerDot + (k - 5)] * 16000 for k in range(11)])
            A = numpy.vstack([x, numpy.ones(len(x))]).T
            [a, b], _, _, _ = numpy.linalg.lstsq(A, currentDots, rcond=None)
            r, p = pearsonr(currentDots, a * x + b)
            slopes.append(a)
            pvalues.append(p)

        axproba.plot(xformant[5:-5], [numpy.arctan(slope) for slope in slopes], 'g', label='Arctan(F2\')')
        axproba.plot(xformant[5:-5], pvalues, 'r', label='p-values of slopes')

    cnnRising = [-100 if neg > pos else 2500 for neg, pos in scores]
    cnnFalling = [500 if neg > pos else -100 for neg, pos in scores]
    pRising = [pos for _, pos in scores]
    xres = numpy.linspace(0.055, len(image[0]) / 16000 - 0.055, len(cnnRising))
    aximg.plot(xres, cnnRising, 'r|', label='Rising')
    aximg.plot(xres, cnnFalling, 'b|    ', label='Falling')
    aximg.set_xlabel("Time(s)")
    aximg.set_ylabel("Frequency(Hz)")
    axproba.set_xlabel("Time(s)")

    axproba.plot(xres, pRising, label='Pro')
    # Plotting the phonemes
    mini = axproba.get_ylim()[0]
    maxi = axproba.get_ylim()[1]
    if phonemes is not None:
        for phoneme, start, end in phonemes:
            axproba.axvline(end / 16000, color="k", linewidth=2)
            axproba.text((end + start) / 32000, mini - 0.12 * (maxi - mini), phoneme, fontsize=8,
                         horizontalalignment='center', verticalalignment='top', weight='bold')

    # GetLabelsFromFile('resources/f2cnn/TEST/DR1.FELC0.SX216.WAV')
    aximg.legend()
    axproba.legend()
    axproba.minorticks_on()
    axproba.grid(True, 'major', linestyle='-')
    axproba.grid(True, 'minor', linestyle='--')
    plt.title(title)
    figMgr=plt.get_current_fig_manager()
    figMgr.resize(*figMgr.window.maxsize())
    # plt.show(fig)
    plt.savefig(os.path.join("graphs", "FallingOrRising", "FallingOrRising." + os.path.split(title)[1] + ".png"),
                dpi=100)
    plt.close(fig)
