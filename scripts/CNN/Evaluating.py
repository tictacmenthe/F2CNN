"""

This file includes functions allowing to evaluate the neural network on files, giving detailed plots of the results.

"""

import glob
import os
import time
from configparser import ConfigParser

import numpy

from gammatone import filters
from scripts.plotting.PlottingCNN import PlotEnvelopesAndCNNResultsWithPhonemes
from scripts.processing.EnvelopeExtraction import ExtractEnvelopeFromMatrix
from scripts.processing.FBFileReader import ExtractFBFile
from scripts.processing.GammatoneFiltering import GetArrayFromWAV, GetFilteredOutputFromArray
from scripts.processing.PHNFileReader import ExtractPhonemes
from .Training import normalizeInput


def EvaluateOneFile(wavFileName, keras, CENTER_FREQUENCIES=None,
                    FILTERBANK_COEFFICIENTS=None):
    """
    Evaluates one .WAV file with the keras model 'last_trained_model'.
    The model should take an input of Nx11x128x1, N being the number of frames in the file, minus the first and last 0.055ms.
    Its output should be two categories, the first one is 'falling', the second 'rising'.
    Produces graphs showing envelope amplitudes, formant frequency if an .FB file is available, results of the model.
    :param keras: variable allowing usage of the keras module
    :param wavFileName: Path to the .WAV file used. If VTR Formants are used, the corresponding .FB file should have the same basename and be in the same directory.
    :param CENTER_FREQUENCIES: (OPTIONAL) Center frequencies of the gammatone filterbank, used for filtering, and also for plotting a spectrogram like figure.
    :param FILTERBANK_COEFFICIENTS: (OPTIONAL) Coefficients of the gammatone filterbank. Should be constructed with the gammatone library's 'gammatone.filters.make_erb.filters' function.
    """

    print("File:\t\t{}".format(wavFileName))

    # #### READING CONFIG FILE
    config = ConfigParser()
    config.read('F2CNN.conf')
    ustos = 1 / 1000000.

    framerate, wavList = GetArrayFromWAV(wavFileName)

    if CENTER_FREQUENCIES is None:
        nchannels = config.getint('FILTERBANK', 'NCHANNELS')
        lowcutoff = config.getint('FILTERBANK', 'LOW')
        # ##### PREPARATION OF FILTERBANK
        # CENTER FREQUENCIES ON ERB SCALE
        CENTER_FREQUENCIES = filters.centre_freqs(framerate, nchannels, lowcutoff)
        # Filter coefficients for a Gammatone filterbank
        FILTERBANK_COEFFICIENTS = filters.make_erb_filters(framerate, CENTER_FREQUENCIES)

    print("Applying filterbank...")
    filtered = GetFilteredOutputFromArray(wavList, FILTERBANK_COEFFICIENTS)
    print("Extracting Envelope...")
    envelopes = ExtractEnvelopeFromMatrix(filtered)
    del filtered

    print("Extracting Formants...")
    fbPath = os.path.splitext(wavFileName)[0] + '.FB'
    formants, sampPeriod = ExtractFBFile(fbPath)

    print("Extracting Phonemes...")
    phnPath = os.path.splitext(wavFileName)[0] + '.PHN'
    phonemes = ExtractPhonemes(phnPath)

    print("Generating input data for CNN...")
    nb = int(len(envelopes[0]) - 0.11 * framerate)
    input_data = numpy.zeros([nb, 11, 128])
    print("INPUT SHAPE:", input_data.shape)
    START = int(0.055 * framerate)
    STEP = int(framerate * sampPeriod * ustos)
    for i in range(0, nb):
        input_data[i] = [[channel[START + i + (k - 5) * STEP] for channel in envelopes] for k in range(11)]
    for i, matrix in enumerate(input_data):
        input_data[i] = normalizeInput(matrix)
    input_data.astype('float32')

    print("Evaluating the data with the pretrained model...")
    model = keras.models.load_model('last_trained_model')
    scores = model.predict(input_data.reshape(nb, 11, 128, 1), verbose=1)
    del model
    del input_data

    print("Plotting...")
    PlotEnvelopesAndCNNResultsWithPhonemes(envelopes, scores, CENTER_FREQUENCIES, phonemes, formants, wavFileName)

    del envelopes
    del phonemes
    keras.backend.clear_session()
    print("\t\t{}\tdone !".format(wavFileName))


def EvaluateRandom(testMode=False):
    import keras
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence tensorflow logs

    TotalTime = time.time()

    if not os.path.isdir("graphs"):
        os.mkdir('graphs')
        os.mkdir(os.path.join('graphs', 'FallingOrRising'))
    if testMode:
        # # Test Files
        wavFiles = glob.glob(os.path.join('testFiles', '*.WAV'))
    else:
        # Get all the WAV files under resources/fcnn
        wavFiles = glob.glob(os.path.join('resources', 'f2cnn', '*', '*.WAV'))
    print("\n###############################\nEvaluating network on WAV {} files in '{}'.".format(len(wavFiles),
                                                                                                  os.path.split(
                                                                                                      wavFiles[0])[0]))

    if not wavFiles:
        print("NO WAV FILES FOUND")
        exit(-1)

    # Reading the config file
    config = ConfigParser()
    config.read('F2CNN.conf')
    framerate = config.getint('FILTERBANK', 'FRAMERATE')
    nchannels = config.getint('FILTERBANK', 'NCHANNELS')
    lowcutoff = config.getint('FILTERBANK', 'LOW')
    # CENTER FREQUENCIES ON ERB SCALE
    CENTER_FREQUENCIES = filters.centre_freqs(framerate, nchannels, lowcutoff)
    FILTERBANK_COEFFICIENTS = filters.make_erb_filters(framerate, CENTER_FREQUENCIES)

    numpy.random.shuffle(wavFiles)
    for file in wavFiles:
        EvaluateOneFile(file, keras, CENTER_FREQUENCIES, FILTERBANK_COEFFICIENTS)
        # We give keras to avoid importing it for every file, and avoid importing it globally as it slows application startup

    print("Evaluating network on all files.")
    print('              Total time:', time.time() - TotalTime)
    print('')
