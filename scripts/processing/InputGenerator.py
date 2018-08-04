import csv
import time
from os.path import join, isdir

import numpy


def GetListOfEnvelopeFilesAndTimepoints(labelFilename):
    """
    Takes a label csv file, and generates a list of [['TEST' or 'TRAIN', filename], [timepoints]] arrays
    :param labelFilename: csv label file
    :return: the described array
    """
    output = dict()
    with open(labelFilename, 'r') as labelFile:
        csvLabelReader = csv.reader(labelFile)
        for i, (testOrTrain, region, speaker, sentence, phoneme, timepoint, slope, pvalue, sign) in enumerate(csvLabelReader):
            file = join(testOrTrain, '.'.join((region, speaker, sentence, 'ENV1.npy')))
            if file not in output.keys():
                output[file]=[timepoint]
            else:
                output[file].append(timepoint)
    return output


def GenerateInputData(testMode):
    STEP = 160
    START = 5 * STEP
    TotalTime = time.time()

    if testMode:
        # Test files trainingData directory
        if not isdir(join("testFiles", "trainingData")):
            print("LABEL GENERATION SHOULD BE DONE PRIOR TO INPUT...")
            exit(-1)
        csvFilename = join("testFiles", "trainingData", "label_data.csv")
    else:
        # The csv label data is inside regular trainingData directory
        if not isdir(join("trainingData")):
            print("LABEL GENERATION SHOULD BE DONE PRIOR TO INPUT...")
            exit(-1)
        csvFilename = join("trainingData", "label_data.csv")

    filesAndTimepointsDict = GetListOfEnvelopeFilesAndTimepoints(csvFilename)

    print("\n###############################\nGenerating Input Data from files with '{}'.".format(csvFilename))

    if not filesAndTimepointsDict:
        print("NO ENV1.npy FILES FOUND")
        exit(-1)

    print(len(envFilesAndTimepoints), "files found along with their",
          sum([len(data[1]) for data in envFilesAndTimepoints]), "entry timepoints.")

    inputData = []
    for i, (file, timepoints) in enumerate(envFilesAndTimepoints):
        if testMode:
            file = join('testFiles', file[1])
        else:
            file = join('resources', 'f2cnn', file[0], file[1])
        print(i, "Reading:\t{}".format(file))
        envelopes = numpy.load(file)

        # Discretisation of the values for each entry required
        currentStep = START
        steps = []
        for t, timepoint in enumerate(timepoints):
            entry = [timepoint + (k - 5) * STEP for k in range(11)]
            currentStep += STEP
            steps.append(entry)
            # print(t, entry)

        for entry in steps:
            entryMatrix = []  # All the values for one entry
            for index in entry:
                valueArray = [channel[index] for channel in envelopes]  # All the values of env at the steps' timepoint
                entryMatrix.append(valueArray)
            inputData.append(entryMatrix)
        print("\t\t{}\tdone !".format(file))

    inputData = numpy.array(inputData, dtype=numpy.float32)
    print('Generated Input Matrix of shape {}.'.format(inputData.shape))

    if testMode:
        savePath = join('testFiles', 'trainingData', 'input_data')
    else:
        savePath = join('trainingData','input_data')
    numpy.save(savePath, inputData)
    print('                Total time:', time.time() - TotalTime)
    print('')
