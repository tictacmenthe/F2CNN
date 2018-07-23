"""

This file allows the reading of VTR Formant database's .FB files, containing the F1,F2,F3 and F3 formants' frequencies
and bandwiths.
The .FB files should be organised like the output of the OrganiseFiles.py script,
inside ../f2cnn/TEST OR TRAIN/ with the names DRr.reader.sentence.FB, with r the regionm reader the ID of the reader
and sentence the ID of the sentence read.
The output is a numpy.ndarray of size 8*nb_frames, with bn_frames being one of the header parameters of the .FB file.

"""

import struct

import numpy


def printBytes(byteStr):
    print(' '.join(hex(nb) for nb in byteStr))
    print('\n')


def ExtractFBFile(fbFilename, verbose=False):
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
        if verbose:
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


def GetF2Frequencies(fbFilename):
    """
    Extracts the F2 formant from data of VTR formants database
    :param fbFilename: path to the file to be used
    :return: array of F2 frequencies in Hz, and the sample period of the file
    """
    matrix, sampPeriod = ExtractFBFile(fbFilename)
    return matrix[:, 1], sampPeriod


def GetF2FrequenciesAround(array, timepoint, radius):
    start, end = timepoint / 160 - radius, timepoint / 160 + radius
    if start < 0 or end >= len(array):
        print("ERROR: WRONG RANGE IN GETF2FREQUENCIESAROUND IN ARRAY:\n", array, "\nAT TIME AND RADIUS", timepoint,
              radius)
        print("INF" if start < 0 else "SUP")
        print(len(array))
        exit(-1)
    start, end = int(start), int(end) + 1
    return array[start:end]
