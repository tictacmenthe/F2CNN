"""

This file allows the reading of VTR Formant database's .FB files, containing the F1,F2,F3 and F3 formants' frequencies and bandwiths.
The .FB files should be organised like the output of the OrganiseFiles.py script,
inside ../f2cnn/TEST OR TRAIN/ with the names DRr.reader.sentence.FB, with r the regionm reader the ID of the reader
and sentence the ID of the sentence read.
The output is a numpy.ndarray of size 8*nb_frames, with bn_frames being one of the header parameters of the .FB file.

"""

import glob
import struct
import time
from multiprocessing.pool import Pool
from os.path import join
import matplotlib.pyplot as plt
import numpy

from scripts.GammatoneFiltering import CENTER_FREQUENCIES


def printBytes(byteStr):
    print(' '.join(hex(nb) for nb in byteStr))
    print('\n')


def ExtractFBFile(fbFilename):
    outputMatrix = None
    sampPeriod = 0

    # The file to read from, in binary reading mode
    with open(fbFilename, 'rb') as fbFile:
        # Reading the headers, with nb of frames and periods
        dat = fbFile.read(4)
        print(dat)
        nFrame = struct.unpack('>i', dat)[0]
        dat = fbFile.read(4)
        sampPeriod = struct.unpack('>i', dat)[0]
        dat = fbFile.read(2)
        sampSize = struct.unpack('>h', dat)[0]
        nComps = sampSize / 4
        dat = fbFile.read(2)
        fileType = struct.unpack('>h', dat)[0]

        print('N_SAMPLES=', nFrame)
        print('SAMP_PERIOD=', sampPeriod)
        print('SAMP_SIZE=', sampSize)
        print('NUM_COMPS=', nComps)
        print('FILE_TYPE=', fileType)

        # The output matrix containing the data of the .FB file without the headers
        outputMatrix = numpy.zeros([nFrame, 8])

        # Reading the values of the formants
        for n in range(nFrame):
            # Read 8 floats(F1 F2 F3 F4 B1 B2 B3 B4) in big endian disposition
            line = fbFile.read(4 * 8)
            data = struct.unpack('>ffffffff', line)
            for i, d in enumerate(data):
                # We want the values in Hz
                outputMatrix[n][i] = round(d * 1000, 2)
    return outputMatrix, sampPeriod


def extractF2Frequencies(fbFilename):
    """
    Extracts the F2 formant from data of VTR formants database
    :param fbFilename: path to the file to be used
    :return: array of F2 frequencies in Hz, and the sample period of the file
    """
    matrix, sampPeriod = ExtractFBFile(fbFilename)
    return matrix[:,1],sampPeriod


def main():
    # # In case you need to print numpy outputs:
    numpy.set_printoptions(threshold=numpy.inf, suppress=True)
    print("Extraction of FB Files...")
    TotalTime = time.time()

    # # Get all the WAV files under ../src
    # fbFiles = glob.glob(join("..","src","f2cnn","*","*.FB"))

    # Test WavFiles
    fbFiles = ["../src/f2cnn/TEST/DR1.FELC0.SI1386.FB"]
    # glob.glob(join("..", "testFiles", "smallest", "*.FB"))[0],
    # glob.glob(join("..", "testFiles", "biggest", "*.FB"))[0]]

    if not fbFiles:
        print("NO FB FILES FOUND")
        exit(-1)
    print(fbFiles)
    print(CENTER_FREQUENCIES)
    ExtractFBFile(fbFiles[0])

    print('              Total time:', time.time() - TotalTime)


if __name__ == '__main__':
    main()