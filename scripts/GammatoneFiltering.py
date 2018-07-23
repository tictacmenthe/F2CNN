"""

This script runs the WAV files from TIMIT database that are needed with VTR FORMANTS through a Gammatone FilterBank,
and saves the output (128*NBFrame floats for each WAV file) to the f2cnn/TRAIN or TEST directories,
with the .GFB.npy extension

The files should be first processed with OrganiseFiles.py

"""

import glob
import struct
import subprocess
import time
import wave
from multiprocessing.pool import Pool
from os import remove
from os.path import splitext, join, split
from shutil import copyfile
import numpy

from gammatone import filters

# ##### PREPARATION OF FILTERBANK
# CENTER FREQUENCIES ON ERB SCALE
CENTER_FREQUENCIES = filters.centre_freqs(16000, 128, 100)
# Filter coefficient for a Gammatone filterbank
FILTERBANK_COEFFICIENTS = filters.make_erb_filters(16000, CENTER_FREQUENCIES)


def getFilteredOutputFromFile(filename):
    """
    Computes the output of a gammatone filterbank applied to the WAV file 'filename'
    :param filename: path to a WAV file
    :return: number of frames in the file, and output matrix (128*nbframes) of the filterbank
    """
    # .WAV file to list
    try:
        wavFile = wave.open(filename, 'r')
    except wave.Error:
        print("Converting file to correct format...")
        convertWavFile(filename)
        wavFile = wave.open(filename, 'r')
    wavList = numpy.zeros(wavFile.getnframes())
    for i in range(wavFile.getnframes()):
        a = wavFile.readframes(1)
        a = struct.unpack("<h", a)[0]
        wavList[i] = a

    # # If plotting is needed
    # t = [i for i in range(len(wavList))]
    # plt.plot(t, wavList)
    # plt.show()

    return getFilteredOutputFromArray(wavFile.getnframes(), wavList)


def getFilteredOutputFromArray(nframes, array):
    # gammatone library needs a numpy array
    # Application of the filterbank to a vector
    filteredMatrix = filters.erb_filterbank(array,
                                            FILTERBANK_COEFFICIENTS)  # Matrix of wavFile.getnframes() X 128 real values
    return nframes, filteredMatrix


def convertWavFile(filename):
    """
    Some WAV files seem to miss some features needed by the wave library (RIFF ID), this counters that
    :param filename: path to the WAV file
    """
    newname = splitext(filename)[0] + '.mp3'
    copyfile(filename, newname)
    remove(filename)
    subprocess.call(['ffmpeg', '-i', newname, filename])
    remove(newname)


def saveGFBMatrix(filename, matrix):
    numpy.save(filename, matrix)


def loadGFBMatrix(filename):
    return numpy.load(filename + '.npy')


def GammatoneFiltering(wavFile):
    gfbFilename = splitext(wavFile)[0] + '.GFB'
    print("Filtering:\t{}".format(wavFile))

    # Compute the filterbank output
    nbframes, outputMatrix = getFilteredOutputFromFile(wavFile)

    # Save file to .GFB.npy format
    print("Saving:\t\t{}".format(gfbFilename))
    saveGFBMatrix(gfbFilename, outputMatrix)
    print("\t\t{} done !".format(wavFile))


def FilterAllOrganisedFiles(testMode):
    TotalTime = time.time()

    if testMode:
        # Test WavFiles
        wavFiles = glob.glob(join("testFiles","*.WAV"))
    else:
        # Get all the WAV files under resources
        wavFiles = glob.glob(join("resources", "f2cnn", "*", "*.WAV"))

    print("\n###############################\nApplying FilterBank to files in '{}'.".format(split(wavFiles[0])[0]))

    if not wavFiles:
        print("NO WAV FILES FOUND")
        exit(-1)

    print(len(wavFiles), "files found")

    # Usage of multiprocessing, to reduce computing time
    proc = 4
    multiproc_pool = Pool(processes=proc)
    multiproc_pool.map(GammatoneFiltering, wavFiles)

    print("Filtered and Saved all files.")
    print('                Total time:', time.time() - TotalTime)
    print('')
