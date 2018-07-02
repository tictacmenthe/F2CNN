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
    outputArray = None
    sampPeriod = 0

    # The file to read from, in binary reading mode
    with open(fbFilename, 'rb') as fbFile:
        # Reading the headers, with nb of frames and periods
        dat = fbFile.read(4)
        nFrame = struct.unpack('>i', dat)[0]
        dat = fbFile.read(4)
        sampPeriod = struct.unpack('>i', dat)[0]
        dat = fbFile.read(2)
        sampSize = struct.unpack('>h', dat)[0]
        nComps = sampSize / 4
        dat = fbFile.read(2)
        fileType = struct.unpack('>h', dat)[0]

        # print('N_SAMPLES=', nFrame)
        # print('SAMP_PERIOD=', sampPeriod)
        # print('SAMP_SIZE=', sampSize)
        # print('NUM_COMPS=', nComps)
        # print('FILE_TYPE=', fileType)

        # The output array, that will contain the F2 column
        outputArray = numpy.zeros(nFrame)

        # Reading the values of the F2 formant
        for n in range(nFrame):
            # Read 8 floats(F1 F2 F3 F4 B1 B2 B3 B4) in big endian disposition
            line = fbFile.read(4 * 8)
            data = struct.unpack('>ffffffff', line)
            # We only get the F2 column, in Hz
            data = data[1] * 1000
            outputArray[n] = data
    return outputArray, sampPeriod


def main():
    # # In case you need to print numpy outputs:
    numpy.set_printoptions(threshold=numpy.inf, suppress=True)
    print("Extraction of FB Files...")
    TotalTime = time.time()

    # # Get all the WAV files under ../src
    # fbFiles = glob.glob(join("..","src","f2cnn","*","*.FB"))

    # Test WavFiles
    fbFiles = "../src/f2cnn/TEST/DR1.FELC0.SI1386.FB"
    # glob.glob(join("..", "testFiles", "smallest", "*.FB"))[0],
    # glob.glob(join("..", "testFiles", "biggest", "*.FB"))[0]]
    print(fbFiles)
    print(CENTER_FREQUENCIES)
    ExtractFBFile(fbFiles)

    print('              Total time:', time.time() - TotalTime)


if __name__ == '__main__':
    main()
