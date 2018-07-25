"""

This file generates labelling data for the CNN, as a .CSV file of columns:
Region(DR1-8),SpeakerID,SentenceID,framepoint,slope,p-valueOfSlope,slopeSign(+-1)
Requires a prior execution of the OrganiseFiles.py, GammatoneFiltering.py, EnvelopeExtraction.py scripts' main functions

"""

import numpy
import csv
import glob
import time
import wave
from os import mkdir
from scipy.stats import pearsonr
from os.path import join, split, splitext, isdir

from .FBFileReader import GetF2Frequencies, GetF2FrequenciesAround
from .PHNFileReader import GetPhonemeAt, VOWELS


def GetLabels(testMode):
    if testMode:
        filepath='testFiles/trainingData/label_data.csv'
    else:
        filepath='trainingData/label_data.csv'
    with open(filepath,'r') as labelDataFile:
        csvReader=csv.reader(labelDataFile)
        return [int(line[-1]) for line in csvReader]


def GenerateLabelData(testMode):
    print(split(__file__)[1])
    STEP = 160
    START = 5 * STEP
    RISK = 0.05  # 5%
    TotalTime = time.time()

    if testMode:
        # Test files
        filenames = glob.glob(join("testFiles", "*.WAV"))
    else:
        # Get all the files under resources
        filenames = glob.glob(join("resources", "f2cnn", "*", "*.WAV"))

    print("\n###############################\nGenerating Label Data from files in '{}'.".format(split(split(filenames[0])[0])[0]))

    # Alphanumeric order
    filenames = sorted(filenames)

    if not filenames:
        print("NO FILES FOUND")
        exit(-1)

    print(len(filenames), "files found")

    vowelCounter = 0
    csvLines = []

    # Get the data into a list of lists for the CSV
    for i, file in enumerate(filenames):
        print(i, "Reading:\t{}".format(file))

        # Get number of points
        wavFile = wave.open(file, 'r')
        nb = int((wavFile.getnframes() / wavFile.getframerate() - 0.11) / 0.01)
        wavFile.close()

        # Get the information about the person
        region, speaker, sentence, _ = split(file)[1].split(".")

        # Discretization of the values for each entry required
        currentStep = START
        steps = []
        for t in range(nb):
            entry = [currentStep + (k - 5) * STEP for k in range(11)]
            currentStep += STEP
            steps.append(entry)

        # Load the F2 values of the file
        F2Array, _ = GetF2Frequencies(splitext(file)[0] + '.FB')

        for step in steps:
            phoneme = GetPhonemeAt(splitext(file)[0] + '.PHN', step[5])
            if phoneme not in VOWELS:
                continue
            entry = [region, speaker, sentence, phoneme, step[5]]
            F2Values = numpy.array(GetF2FrequenciesAround(F2Array, step[5], 5))

            # Least Squares Method for linear regression of the F2 values
            x = numpy.array(step)
            A = numpy.vstack([x, numpy.ones(len(x))]).T
            [a, b], _, _, _ = numpy.linalg.lstsq(A, F2Values, rcond=None)

            # Pearson Correlation Coefficient r and p-value p using scipy.stats.pearsonr
            r, p = pearsonr(F2Values, a * x + b)

            # We round them up at 5 digits after the comma
            entry.append(round(a, 5))
            entry.append(round(p, 5))
            # The line to be added to the CSV file, only if the direction of the formant is clear enough (% risk)
            if p < RISK:
                vowelCounter += 1
                entry.append(1 if a > 0 else -1)
                csvLines.append(entry)
                # print(r ** 2, p)
                # output = a * x + b
                # print(a, b)
                # fig = pyplot.figure()
                # ax = fig.add_subplot(111)
                #
                # dots, = ax.plot(x, F2Values, 'r.')
                # regress, = ax.plot(x, output)
                # ax.legend((regress, dots), ("Least Squares", "Raw F2 Values"))
                # pyplot.title("Regression of the F2 values +-50ms around {}th frame".format(step[5]))
                # valuesString = "r^2={}, p={}, a={}".format(round(r ** 2, 5), round(p, 5), a)
                # ax.text(-0.1, -0.1, valuesString, transform=ax.transAxes)
                # ax.text(0.8, -0.1, "File: {}".format(file.split("/")[-1]), transform=ax.transAxes)
                # pyplot.show(fig)
        print("\t\t{}\tdone !".format(file))

    # Saving into a file
    if testMode:
        if not isdir(join("testFiles", "trainingData")):
            mkdir(join("testFiles", "trainingData"))
        filePath = join("testFiles", "trainingData", "label_data.csv")
    else:
        filePath = join("trainingData", "label_data.csv")

    with open(filePath, "w") as outputFile:
        writer = csv.writer(outputFile)
        for line in csvLines:
            writer.writerow(line)
    print("Generated Label Data CSV of", vowelCounter, "(vowels only) lines.")
    print('                Total time:', time.time() - TotalTime)
    print('')
