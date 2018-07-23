import glob
import time
from os.path import join, split
import numpy


def GenerateInputData(testMode):
    SAMPLING_FREQ = 16000.
    STEP = 160
    START = 5 * STEP
    TotalTime = time.time()

    if testMode:
        # Test files
        envFiles = glob.glob(join("testFiles", "*.ENV1.npy"))
    else:
        # Get all the ENV1.npy files under resources
        envFiles = glob.glob(join("resources", "f2cnn", "*", "*.ENV1.npy"))
    print("\n###############################\nGenerating Input Data from files in '{}'.".format(split(envFiles[0])[0]))

    envFiles = sorted(envFiles)

    if not envFiles:
        print("NO ENV1.npy FILES FOUND")
        exit(-1)

    print(len(envFiles), "files found")

    inputData = []
    for i, file in enumerate(envFiles):
        print(i, "Reading:\t{}".format(file))
        envelopes = numpy.load(file)

        # Number of timepoints for measures
        nb = int((len(envelopes[0]) / SAMPLING_FREQ - 0.11) / 0.01)

        # Discretisation of the values for each entry required
        currentStep = START
        steps = []
        for t in range(nb):
            entry = [currentStep + (k - 5) * STEP for k in range(11)]
            currentStep += STEP
            steps.append(entry)

        for entry in steps:
            entryMatrix = []  # All the values for one entry
            for index in entry:
                valueArray = [channel[index] for channel in envelopes]  # All the values of env at the steps' timepoint
                entryMatrix.append(valueArray)
            inputData.append(entryMatrix)
        print("\t\t{} done !".format(file))


    inputData = numpy.array(inputData, dtype=numpy.float32)
    print('Generated Input Matrix of shape {}.'.format(inputData.shape))
    numpy.save("trainingData/input_data", inputData)
    print('                Total time:', time.time() - TotalTime)
    print('')
