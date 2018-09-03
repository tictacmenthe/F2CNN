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


def ExtractFBFile(fbFilename, verbose=False):
    # The file to read from, in binary reading mode
    try:
        with open(fbFilename, 'r+b') as fbFile:
            # Reading the headers, with nb of frames and periods
            dat = fbFile.read(4)
            nFrame = struct.unpack('>i', dat)[0]
            dat = fbFile.read(4)
            sampPeriod = 10000  # struct.unpack('>i', dat)[0]
            # TODO: one of the VTR database files has an unusual sampling period (100 000 not 10 000)
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
    except FileNotFoundError:
        print("No .FB formant data file.")
        return None, 0


def GetFormantFrequencies(fbFilename, formant):
    """
    Extracts de formant F_formant's frequencies from data of VTR formants database
    :param fbFilename:  path to the .FB file used
    :param formant: index of the formant(1-4)
    :return: Array of F_formant frequencies in Hz, and the sample period of the file
    """
    matrix, sampPeriod = ExtractFBFile(fbFilename)
    if matrix is not None:
        return matrix[:,formant-1], sampPeriod
    else:
        return None, None


def GetFromantFrequenciesAround(array, timepoint, radius, wavToFormant):
    """
    Gets the values of frequencies around a frame in a formant data Array
    :param array: array of values built with function GetFormantFrequencies
    :param timepoint: center frame to consider
    :param radius: radius used to get values at timepoint+-radius
    :param wavToFormant: ratio of framerate of wav file and sampling rate of formant
    :return: array of radius*2 + 1 frequencies
    """
    start, end = timepoint / wavToFormant - radius, timepoint / wavToFormant + radius
    start, end = int(start), int(end) + 1
    if start < 0 or end >= len(array):
        print("ERROR: WRONG RANGE IN GETFORMANTFREQUENCIESAROUND IN ARRAY OF LEN:\n", len(array), "\nAT TIME AND RADIUS", timepoint,
              radius,"START",start,"END",end)
        if start<0:
            print("INF")
        elif end>=len(array):
            print("SUP")
        print(len(array))
        exit(-1)
    return array[start:end]
