"""

This file includes functions allowing to evaluate the neural network on files, giving detailed plots of the results.

"""

import glob
import os
import time
from configparser import ConfigParser
from shutil import copyfile

import numpy
from matplotlib import pyplot
from scipy.io import wavfile

from gammatone import filters
from scripts.plotting.PlottingCNN import PlotEnvelopesAndCNNResultsWithPhonemes
from scripts.processing.EnvelopeExtraction import ExtractEnvelopeFromMatrix
from scripts.processing.FBFileReader import ExtractFBFile
from scripts.processing.GammatoneFiltering import GetArrayFromWAV, GetFilteredOutputFromArray
from scripts.processing.LabelDataGenerator import ExtractLabel
from scripts.processing.PHNFileReader import ExtractPhonemes
from .Training import normalizeInput


def EvaluateOneWavArray(wavArray, framerate, wavFileName, model='last_trained_model', LPF=False, CUTOFF=100,CENTER_FREQUENCIES=None,
                        FILTERBANK_COEFFICIENTS=None):
    # #### READING CONFIG FILE
    config = ConfigParser()
    config.read('F2CNN.conf')
    RADIUS = config.getint('CNN', 'RADIUS')
    SAMPPERIOD = config.getint('CNN', 'SAMPLING_PERIOD')
    NCHANNELS = config.getint('FILTERBANK', 'NCHANNELS')
    DOTSPERINPUT = RADIUS * 2 + 1
    USTOS = 1 / 1000000.

    # Extracting labels, for accuracy computation
    labels = ExtractLabel(wavFileName, config)
    labels = [(entry[-4], entry[-1]) for entry in labels] if labels is not None else None

    if CENTER_FREQUENCIES is None:
        NCHANNELS = config.getint('FILTERBANK', 'NCHANNELS')
        lowcutoff = config.getint('FILTERBANK', 'LOW_FREQUENCY')
        # ##### PREPARATION OF FILTERBANK
        # CENTER FREQUENCIES ON ERB SCALE
        CENTER_FREQUENCIES = filters.centre_freqs(framerate, NCHANNELS, lowcutoff)
        # Filter coefficients for a Gammatone filterbank
        FILTERBANK_COEFFICIENTS = filters.make_erb_filters(framerate, CENTER_FREQUENCIES)

    print("Applying filterbank...")
    filtered = GetFilteredOutputFromArray(wavArray, FILTERBANK_COEFFICIENTS)
    del wavArray
    if not LPF:
        print("Extracting Envelope...")
    else:
        print("Extraction Envelope with {}Hz Low Pass Filter...".format(CUTOFF))
    envelopes = ExtractEnvelopeFromMatrix(filtered, LPF, CUTOFF)
    del filtered

    print("Extracting Formants...")
    fbPath = os.path.splitext(wavFileName)[0] + '.FB'
    formants, sampPeriod = ExtractFBFile(fbPath)

    print("Extracting Phonemes...")
    phnPath = os.path.splitext(wavFileName)[0] + '.PHN'
    phonemes = ExtractPhonemes(phnPath)

    print("Generating input data for CNN...")
    STEP = int(framerate * SAMPPERIOD * USTOS)
    START = int(STEP * RADIUS)
    nb = int(len(envelopes[0]) - DOTSPERINPUT*STEP)
    input_data = numpy.zeros([nb, DOTSPERINPUT, NCHANNELS])
    print("INPUT SHAPE:", input_data.shape)
    for i in range(0, nb):
        input_data[i] = [[channel[START + i + (k - RADIUS) * STEP] for channel in envelopes] for k in
                         range(DOTSPERINPUT)]
    for i, matrix in enumerate(input_data):
        input_data[i] = normalizeInput(matrix)
    input_data.astype('float32')

    print("Evaluating the data with the pretrained model...")
    import keras
    model = keras.models.load_model(model)
    scores = model.predict(input_data.reshape(nb, DOTSPERINPUT, NCHANNELS, 1), verbose=1)
    simplified_scores = [1 if score[1] > score[0] else 0 for score in scores]
    # Attempt to compute an accuracy for the file. TODO: Doesn't take into account phonemes we use, step values
    keras.backend.clear_session()
    del model
    del input_data
    accuracy = None
    if labels is not None:
        accuracy = 0
        total_valid = 0
        for timepoint, score in enumerate(simplified_scores):
            for index in range(len(labels) - 1):
                before = labels[index][0]
                after = labels[index + 1][0]
                if before < timepoint < after and (abs(timepoint - before) < STEP or abs(timepoint - after) < STEP):
                    if abs(before - timepoint) <= abs(after - timepoint):
                        if score == labels[index][1]:
                            accuracy += 1
                    else:
                        if score == labels[index + 1][1]:
                            accuracy += 1
                    total_valid += 1
        accuracy /= total_valid
    print("Plotting...")
    PlotEnvelopesAndCNNResultsWithPhonemes(envelopes, scores, accuracy, CENTER_FREQUENCIES, phonemes, formants,
                                           wavFileName)
    del envelopes
    del phonemes


def EvaluateOneWavFile(file, LPF=False, CUTOFF=50, model='last_trained_model', CENTER_FREQUENCIES=None,
                       FILTERBANK_COEFFICIENTS=None):
    """
    Evaluates one .WAV file with the keras model 'last_trained_model'.
    The model should take an input of Nx11x128x1, N being the number of frames in the file, minus the first and last 0.055ms.
    Its output should be two categories, the first one is 'falling', the second 'rising'.
    Produces graphs showing envelope amplitudes, formant frequency if an .FB file is available, results of the model.
    :param file: Path to the evaluated file
    :param LPF: Boolean specifying if using low pass filtering for envelope extraction
    :param CUTOFF: Low Pass Filter cutoff frequency
    :param model: the keras model file to use
    :param CENTER_FREQUENCIES: (OPTIONAL) Center frequencies of the gammatone filterbank, used for filtering, and also for plotting a spectrogram like figure.
    :param FILTERBANK_COEFFICIENTS: (OPTIONAL) Coefficients of the gammatone filterbank. Should be constructed with the gammatone library's 'gammatone.filters.make_erb.filters' function.
    """
    print('Using model', model)
    print("File:\t\t{}".format(file))
    framerate, wavArray = GetArrayFromWAV(file)
    EvaluateOneWavArray(wavArray=wavArray, framerate=framerate, LPF=LPF, CUTOFF=CUTOFF,wavFileName=file, model=model,
                        CENTER_FREQUENCIES=CENTER_FREQUENCIES, FILTERBANK_COEFFICIENTS=FILTERBANK_COEFFICIENTS)
    print("\t\t{}\tdone !".format(file))


def EvaluateRandom(count=None, LPF=False, CUTOFF=50):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence tensorflow logs

    TotalTime = time.time()

    if not os.path.isdir("graphs"):
        os.mkdir('graphs')
        os.mkdir(os.path.join('graphs', 'FallingOrRising'))

    # Get all the WAV files under resources/fcnn
    wavFiles = glob.glob(os.path.join('resources', 'f2cnn', '*', '*.WAV'))
    print("\n###############################\nEvaluating network on {} WAV files in '{}'.".format(len(wavFiles),
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
    lowcutoff = config.getint('FILTERBANK', 'LOW_FREQUENCY')
    # CENTER FREQUENCIES ON ERB SCALE
    CENTER_FREQUENCIES = filters.centre_freqs(framerate, nchannels, lowcutoff)
    FILTERBANK_COEFFICIENTS = filters.make_erb_filters(framerate, CENTER_FREQUENCIES)

    # Selecting some random files, or all of them
    if count is None:
        numpy.random.shuffle(wavFiles)
    elif count > 1:
        wavFiles = numpy.random.choice(wavFiles, count)

    for file in wavFiles:
        EvaluateOneWavFile(file, LPF=LPF, CUTOFF=CUTOFF, CENTER_FREQUENCIES=CENTER_FREQUENCIES, FILTERBANK_COEFFICIENTS=FILTERBANK_COEFFICIENTS)

    print("Evaluating network on all files.")
    print('              Total time:', time.time() - TotalTime)
    print('')


def SNRdbToSNRlinear(SNRdb):
    return 10 ** (SNRdb / 10.0)


def RMS(signal):
    """
    Computes the Root Mean Square Error of a signal
    :param signal:
    :return:
    """
    return numpy.sqrt(numpy.mean(numpy.square(signal)))


def EvaluateWithNoise(file, LPF=False, CUTOFF=100, model='last_trained_model', CENTER_FREQUENCIES=None,
                      FILTERBANK_COEFFICIENTS=None, SNRdB=-3):
    print("File:\t\t{}".format(file))
    print("Appyling gaussian noise, new SNR is {SNR}dB".format(SNR=SNRdB))
    framerate, wavList = GetArrayFromWAV(file)
    # Generating noise
    noise = numpy.random.normal(scale=RMS(wavList) / SNRdbToSNRlinear(SNRdB), size=wavList.shape[0])
    output = noise + wavList

    # Creating and saving the new wav file
    os.makedirs(os.path.join('OutputWavFiles', 'addedNoise'), exist_ok=True)
    baseName = os.path.join('OutputWavFiles', 'addedNoise',
                            os.path.split(os.path.splitext(file)[0])[1]) + '{SNR}dB'.format(SNR=SNRdB)
    newPath = baseName + '.WAV'
    srcBasename = os.path.splitext(file)[0]

    wavfile.write(newPath, framerate, output)
    try:
        copyfile(srcBasename + '.FB', baseName + '.FB')
        copyfile(srcBasename + '.PHN', baseName + '.PHN')
        copyfile(srcBasename + '.WRD', baseName + '.WRD')
    except FileNotFoundError as e:
        print(e.strerror)
        print("No .FB or .PHN or .WRD files.")

    print('New noisy WAVE file saved as', newPath)
    EvaluateOneWavArray(output, framerate, newPath, model=model, LPF=LPF, CUTOFF=CUTOFF, CENTER_FREQUENCIES=CENTER_FREQUENCIES, FILTERBANK_COEFFICIENTS=FILTERBANK_COEFFICIENTS)

    print("\t\t{}\tdone !".format(file))
