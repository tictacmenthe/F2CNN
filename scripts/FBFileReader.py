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
    outputList = []
    with open(fbFilename, 'rb') as fbFile:
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
        array = numpy.zeros(nFrame)
        for n in range(nFrame):
            line = fbFile.read(4 * 8)
            data = struct.unpack('>ffffffff', line)
            data = data[1]*1000
            array[n] = data
        return array


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
    print(fbFiles)
    print(CENTER_FREQUENCIES)
    ExtractFBFile(fbFiles[0])
    # # Usage of multiprocessing, to reduce computing time
    # proc = 4
    # multiproc_pool = Pool(processes=proc)
    # multiproc_pool.map(ExtractFBFile, fbFiles)

    print('              Total time:', time.time() - TotalTime)


if __name__ == '__main__':
    main()
