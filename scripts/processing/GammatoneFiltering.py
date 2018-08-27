"""

This script runs the WAV files from TIMIT database that are needed with VTR FORMANTS through a Gammatone FilterBank,
and saves the output (128*NBFrame floats for each WAV file) to the f2cnn/TRAIN or TEST directories,
with the .GFB.npy extension

The files should be first processed with OrganiseFiles.py

"""

import glob
import time
from configparser import ConfigParser
from itertools import repeat
from multiprocessing import cpu_count, Value
from multiprocessing.pool import Pool
from os.path import splitext, join, split

import numpy
from scipy.io import wavfile as WavFileTool
from sphfile import SPHFile

from gammatone import filters

counter = None


def GetArrayFromWAV(filename):
    with open(filename, 'rb') as wavFile:
        header=wavFile.read(4)
    if header == b'RIFF':   # RIFF header, for WAVE files
        framerate, wavArray = WavFileTool.read(filename)
    else:                   # NIST header, which uses SPHERE
        file=SPHFile(filename)
        framerate = file.format['sample_rate']
        wavArray=numpy.zeros(len(file.time_range()), dtype=numpy.int16)
        for i, value in enumerate(file.time_range()):
            wavArray[i]=value
    return framerate, wavArray


def GetFilteredOutputFromArray(array, FILTERBANK_COEFFICIENTS):
    # gammatone library needs a numpy array
    # Application of the filterbank to a vector
    filteredMatrix = filters.erb_filterbank(array, FILTERBANK_COEFFICIENTS)
    # Matrix of wavFile.getnframes() X 128 real values
    return filteredMatrix


def GetFilteredOutputFromFile(filename, FILTERBANK_COEFFICIENTS):
    """
    Computes the output of a gammatone filterbank applied to the WAV file 'filename'
    :param FILTERBANK_COEFFICIENTS
    :param filename: path to a WAV file
    :return: number of frames in the file, and output matrix (128*nbframes) of the filterbank
    """
    framerate, wavArray = GetArrayFromWAV(filename)
    return GetFilteredOutputFromArray(wavArray, FILTERBANK_COEFFICIENTS), framerate


def saveGFBMatrix(filename, matrix):
    numpy.save(filename, matrix)


def loadGFBMatrix(filename):
    return numpy.load(filename + '.npy')


def GammatoneFiltering(wavFile, n):
    gfbFilename = splitext(wavFile)[0] + '.GFB'
    print("Filtering:\t{}".format(wavFile))

    # Compute the filterbank output
    outputMatrix, _ = GetFilteredOutputFromFile(wavFile, FILTERBANK_COEFFICIENTS)

    # Save file to .GFB.npy format
    print("Saving:\t\t{}.npy".format(gfbFilename))
    saveGFBMatrix(gfbFilename, outputMatrix)

    global counter
    with counter.get_lock():
        counter.value += 1
        print("\t\t{:<50} done ! {}/{} Files.".format(wavFile, counter.value, n))


def InitProcesses(FBCOEFS, cn):
    global FILTERBANK_COEFFICIENTS
    global counter
    counter = cn
    FILTERBANK_COEFFICIENTS = FBCOEFS


def FilterAllOrganisedFiles():
    TotalTime = time.time()

    # Get all the WAV files under resources
    # wavFiles = glob.glob(join("resources", "f2cnn", "*", "*.WAV"))
    wavFiles = glob.glob(join("resources", "f2cnn", "**", "*.WAV"))

    print("\n###############################\nApplying FilterBank to files in '{}'.".format(split(wavFiles[0])[0]))

    if not wavFiles:
        print("NO WAV FILES FOUND, PLEASE ORGANIZE FILES")
        exit(-1)

    print(len(wavFiles), "files found")

    # #### READING CONFIG FILE
    config = ConfigParser()
    config.read('F2CNN.conf')
    framerate = config.getint('FILTERBANK', 'FRAMERATE')
    nchannels = config.getint('FILTERBANK', 'NCHANNELS')
    lowcutoff = config.getint('FILTERBANK', 'LOW')
    # ##### PREPARATION OF FILTERBANK
    # CENTER FREQUENCIES ON ERB SCALE
    CENTER_FREQUENCIES = filters.centre_freqs(framerate, nchannels, lowcutoff)
    # Filter coefficient for a Gammatone filterbank
    FILTERBANK_COEFFICIENTS = filters.make_erb_filters(framerate, CENTER_FREQUENCIES)

    # Usage of multiprocessing, to reduce computing time
    proc = cpu_count()
    counter = Value('i', 0)
    multiproc_pool = Pool(processes=proc, initializer=InitProcesses, initargs=(FILTERBANK_COEFFICIENTS, counter,))
    multiproc_pool.starmap(GammatoneFiltering, zip(wavFiles, repeat(len(wavFiles))))

    print("Filtered and Saved all files.")
    print('                Total time:', time.time() - TotalTime)
    print('')
