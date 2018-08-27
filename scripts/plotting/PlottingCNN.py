"""
    This file gives access to plotting CNN result data for this project
"""

from configparser import ConfigParser
from os.path import join, split

import matplotlib.pyplot as plt
import numpy
from matplotlib.colors import LogNorm
from scipy.stats import pearsonr

from scripts.plotting.PlottingProcessing import ReshapeEnvelopesForSpectrogram


def PlotEnvelopesAndCNNResultsWithPhonemes(envelopes, scores, CENTER_FREQUENCIES, phonemes, Formants=None, title=""):
    image = ReshapeEnvelopesForSpectrogram(envelopes, CENTER_FREQUENCIES)
    formant = []
    # fig = plt.figure()
    fig = plt.figure(figsize=(32,16))
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

    # Extraction and plotting of rising/falling results
    cnnRising, cnnFalling, cnnNone, pRising = [], [], [], []
    if len(scores[0]) == 2:     # If we only have Rising and Falling classes
        cnnRising = [2500 if pos > neg else -100 for neg, pos in scores]
        cnnFalling = [500 if neg > pos else -100 for neg, pos in scores]
        pRising = [pos for _, pos in scores]
    elif len(scores[0]) == 3:   # If we have a third 'none' class
        cnnRising = [2500 if pos > neg and pos > none else -100 for neg, pos, none in scores]
        cnnFalling = [500 if neg > pos and neg > none else -100 for neg, pos, none in scores]
        cnnNone = [1500 if none > neg and none > pos else -100 for neg, pos, none in scores]
        pRising = [pos for _, pos, _ in scores]
    xres = numpy.linspace(0.055, len(image[0]) / 16000 - 0.055, len(cnnRising))
    aximg.plot(xres, cnnRising, 'r|', label='Rising')
    aximg.plot(xres, cnnFalling, 'b|', label='Falling')
    if len(scores[0]) == 3:
        aximg.plot(xres, cnnNone, 'k|', label='Neither')

    aximg.set_xlabel("Time(s)")
    aximg.set_ylabel("Frequency(Hz)")
    axproba.set_xlabel("Time(s)")

    # Plotting the probability of rising according to the used network
    axproba.plot(xres, pRising, label='Probability of F2 rising for network')

    # Plotting the phonemes
    mini = axproba.get_ylim()[0]
    maxi = axproba.get_ylim()[1]
    if phonemes is not None:
        for phoneme, start, end in phonemes:
            axproba.axvline(end / 16000, color="xkcd:olive", linewidth=1)
            axproba.text((end + start) / 32000, mini - 0.12 * (maxi - mini), phoneme, fontsize=8,
                         horizontalalignment='center', verticalalignment='top', weight='bold')

    # GetLabelsFromFile('resources/f2cnn/TEST/DR1.FELC0.SX216.WAV')
    aximg.legend()
    axproba.legend()
    axproba.minorticks_on()
    axproba.grid(True, 'major', linestyle='-')
    axproba.grid(True, 'minor', linestyle='--')

    # Probability limits
    axproba.axhline(0)
    axproba.axhline(0.5)
    axproba.axhline(1.0)
    xlim=axproba.get_xlim()

    plt.annotate('Rising', xy=(0,1.0), xytext=(-0.05*xlim[1],1.1), arrowprops=dict(facecolor='black', shrink=0.01))
    plt.annotate('Falling', xy=(0,0.0), xytext=(-0.05*xlim[1],-0.1), arrowprops=dict(facecolor='black', shrink=0.01))

    plt.title(title)
    figMgr = plt.get_current_fig_manager()
    figMgr.resize(*figMgr.window.maxsize())
    # plt.show(fig)
    plt.savefig(join("graphs", "FallingOrRising", "FallingOrRising." + split(title)[1] + ".png"),
                dpi=100)
    plt.close(fig)
