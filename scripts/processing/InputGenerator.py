import csv
import glob
import time
from configparser import ConfigParser
from os.path import join, isdir

import numpy
from matplotlib.colors import LogNorm

from scripts.processing.OrganiseFiles import completeSplit


def GetListOfEnvelopeFilesAndTimepoints(labelFilename):
    """
    Takes a label csv file, and generates a list of [['TEST' or 'TRAIN', filename], [timepoints]] arrays
    :param labelFilename: csv label file
    :return: the described array
    """
    output = dict()
    with open(labelFilename, 'r') as labelFile:
        csvLabelReader = csv.reader(labelFile)
        for i, (testOrTrain, region, speaker, sentence, phoneme, timepoint, slope, pvalue, sign) in enumerate(
                csvLabelReader):
            file = join(testOrTrain, '.'.join((region, speaker, sentence, 'ENV1.npy')))
            if file not in output.keys():
                output[file] = [int(timepoint)]
            else:
                output[file].append(int(timepoint))
    return output


def GenerateInputData(LPF=False, CUTOFF=100):
    TotalTime = time.time()

    # The csv label data is inside regular trainingData directory
    if not isdir(join("trainingData")):
        print("LABEL GENERATION SHOULD BE DONE PRIOR TO INPUT...")
        exit(-1)
    csvFilename = join("trainingData", "label_data.csv")

    # Extract all filepaths and timepoints for each file as a dict{file:[timepoints]}
    filesAndTimepointsDict = GetListOfEnvelopeFilesAndTimepoints(csvFilename)

    print("\n###############################\nGenerating Input Data from files with '{}'.".format(csvFilename))
    if LPF:
        print("Using Low Pass Filtering with a cutoff at {}Hz".format(CUTOFF))
    else:
        print("Not using Low Pass Filtering")

    if not filesAndTimepointsDict:
        print("NO ENV1.npy FILES FOUND, PLEASE GENERATE ENVELOPES")
        exit(-1)
    files=filesAndTimepointsDict.keys()
    files=sorted(files)
    totalTimePoints = sum([len(data) for data in filesAndTimepointsDict.values()])
    print(len(filesAndTimepointsDict.keys()), "files found along with their",
          totalTimePoints, "entry timepoints.")
    # #### READING CONFIG FILE
    config = ConfigParser()
    config.read('F2CNN.conf')
    RADIUS = config.getint('CNN', 'RADIUS')
    SAMPPERIOD = config.getint('CNN', 'sampperiod')
    FRAMERATE = config.getint('FILTERBANK', 'framerate')
    NCHANNELS = config.getint('FILTERBANK', 'nchannels')
    DOTSPERINPUT = RADIUS * 2 + 1

    inputData = numpy.zeros((totalTimePoints, DOTSPERINPUT, NCHANNELS))
    print("Output shape:", inputData.shape)
    STEP = int(FRAMERATE*SAMPPERIOD/1000000)
    currentEntry = 0
    for currentFileIndex, file in enumerate(files):
        timepoints=filesAndTimepointsDict[file]
        file = join('resources', 'f2cnn', file)
        print("Reading:\t{}".format(file))
        envelopes = numpy.load(file)
        i=0
        for i, center in enumerate(timepoints):
            entryMatrix = numpy.zeros((DOTSPERINPUT,NCHANNELS))  # All the values for one entry(11 timepoints centered around center) : 11x128 matrix
            for j, index in enumerate([center + STEP * (k - RADIUS) for k in range(DOTSPERINPUT)]):
                valueArray = numpy.array([channel[index] for channel in envelopes])  # All the values of env at the steps' timepoint
                entryMatrix[j]=valueArray
            inputData[currentEntry+i]=entryMatrix
        currentEntry+=i+1
        print("\t\t{:<50} done !  {}/{} Files".format(file,currentFileIndex+1, len(filesAndTimepointsDict.keys())))
    inputData = numpy.array(inputData, dtype=numpy.float32)
    print('Generated Input Matrix of shape {}.'.format(inputData.shape))

    savePath = join('trainingData', 'input_data_LPF{}.npy'.format(CUTOFF) if LPF else 'input_data.npy')

    print("Saving {}...".format(savePath))
    numpy.save(savePath, inputData)
    numpy.save(join('trainingData','last_input_data.npy'), inputData) # make a backup of the last generated, just in case
    print('                Total time:', time.time() - TotalTime)
    print('')
