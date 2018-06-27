"""

This script runs the WAV files from TIMIT database that are needed with VTR FORMANTS through a Gammatone FilterBank,
and saves the output (128*NBFrame floats for each WAV file) to the f2cnn/TRAIN or TEST directorty,
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
from os.path import splitext
from shutil import copyfile

import numpy

from libs.gammatone import filters

# ##### PREPARATION OF FILTERBANK
# CENTER FREQUENCIES ON ERB SCALE
freqs = filters.centre_freqs(16000, 128, 100)
# Filter coefficient for a Gammatone filterbank
filterBankCoeffs = filters.make_erb_filters(16000, freqs)


def getFilteredOutput(filename):
    """
    Computes the output of a gammatone filterbank applied to the WAV file 'filename'
    :param filename: path to a WAV file
    :return: number of frames in the file, and output matrix (128*nbframes) of the filterbank
    """
    # .WAV file to list
    try:
        wavFile = wave.open(filename, 'r')
    except wave.Error:
        print("Converting Files to correct format...")
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

    # gammatone library needs a numpy array

    # Application of the filterbank to a vector
    filteredMatrix = filters.erb_filterbank(wavList,
                                            filterBankCoeffs)  # Matrix of wavFile.getnframes() X 128 real values
    return wavFile.getnframes(), filteredMatrix


def convertWavFile(filename):
    """
    Some WAV files seem to miss some features needed by the wave library (RIFF ID), this counters that
    :param filename: path to the WAV file
    """
    newname = splitext(filename)[0] + '.mp3'
    copyfile(filename, newname)
    remove(filename)
    subprocess.call(['ffmpeg', '-i', newname, filename])


def saveGFBMatrix(filename, matrix):
    numpy.save(filename, matrix)


def loadGFBMatrix(filename):
    return numpy.load(filename + '.npy')


def GammatoneFiltering(wavFile):
    gfbFilename = splitext(wavFile)[0] + '.GFB'

    # Compute the filterbank output
    startTime = time.time()
    nbframes, outputMatrix = getFilteredOutput(wavFile)
    duration = time.time() - startTime
    print(gfbFilename)
    print('        Time for filtering:', duration)

    # # If needed, plotting the first output of the file
    # t = [i for i in range(nbframes)]
    # plt.plot(t,outputMatrix[0])
    # plt.show()

    # Save file to .GFB.npy format
    startTime = time.time()
    saveGFBMatrix(gfbFilename, outputMatrix)
    duration = time.time() - startTime
    print(gfbFilename)
    print('           Time for saving:', duration)


def main():
    # # In case you need to print numpy outputs:
    # numpy.set_printoptions(threshold=numpy.inf, suppress=True)
    TotalTime = time.time()

    # Get all the WAV files under ../src
    wavFiles = glob.glob('../src/f2cnn/*/*.WAV')

    # # Test WavFiles
    # wavFiles = ['../testFiles/testSmall.WAV', '../testFiles/testBig.WAV']

    # Usage of multiprocessing, to reduce computing time
    proc = 4
    multiproc_pool = Pool(processes=proc)
    multiproc_pool.map(GammatoneFiltering, wavFiles)

    print('                Total time:', time.time() - TotalTime)


if __name__ == '__main__':
    main()
