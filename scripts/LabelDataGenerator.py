"""

This file generates labelling data for the CNN, as a .CSV file of columns:
Region(DR1-8),SpeakerID,SentenceID,framepoint,slope,p-valueOfSlope,slopeSign(+-1)
Requires a prior execution of the OrganiseFiles.py, GammatoneFiltering.py, EnvelopeExtraction.py scripts' main functions

"""

import csv
import glob
import time
import wave
from os.path import join, split, splitext
import numpy
from scipy.stats import pearsonr

from .FBFileReader import GetF2Frequencies, GetF2FrequenciesAround
from .PHNFileReader import GetPhonemeAt

vowels = ["iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao",
          "oy", "ow", "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h"]


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

    # Alphanumeric order
    filenames = sorted(filenames)

    if not filenames:
        print("NO FILES FOUND")
        exit(-1)

    print(filenames)

    vowelCounter = 0
    csvLines = []

    # Get the data into a list of lists for the CSV
    for i, file in enumerate(filenames):
        print(i, file)

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
            if phoneme not in vowels:
                continue
            vowelCounter += 1
            entry = [region, speaker, sentence, phoneme, step[5]]
            F2Values = numpy.array(GetF2FrequenciesAround(F2Array, step[5], 5))

            # Least Squares Method for linear regression of the F2 values
            x = numpy.array([i for i in range(11)])
            A = numpy.vstack([x, numpy.ones(len(x))]).T
            [a, b], _, _, _ = numpy.linalg.lstsq(A, F2Values, rcond=None)

            # Pearson Correlation Coefficient r and p-value p using scipy.stats.pearsonr
            r, p = pearsonr(F2Values, a * x + b)

            # We round them up at 5 digits after the comma
            entry.append(round(a, 6))
            entry.append(round(p, 6))

            # The line to be added to the CSV file, only if the direction of the formant is clear enough (% risk)
            if p < RISK:
                entry.append(1 if a > 0 else -1)
                csvLines.append(entry)
                # print(r ** 2, p)
                # output = a * x + b
                # print(a, b)
                # fig = plt.figure()
                # ax = fig.add_subplot(111)
                #
                # dots, = ax.plot(x, F2Values, 'r.')
                # regress, = ax.plot(x, output)
                #
                # ax.legend((regress, dots), ("Least Squares", "Raw F2 Values"))
                # plt.title("Regression of the F2 values +-50ms around {}th frame".format(step[5]))
                # valuesString = "r^2 = {}, p-value = {}".format(round(r ** 2, 5), round(p, 5))
                # ax.text(-0.1, -0.1, valuesString, transform=ax.transAxes)
                # ax.text(0.5, -0.1, "File: {}".format(file.split("/")[-1]), transform=ax.transAxes)
                # plt.show(fig)

    # Saving into a file
    if testMode:
        filePath=join("testFiles","trainingData", "label_data.csv")
    else:
        filePath=join("trainingData", "label_data.csv")

    with open(filePath, "w") as outputFile:
        writer = csv.writer(outputFile)
        for line in csvLines:
            writer.writerow(line)
    print(vowelCounter, "vowels found")
    print('                Total time:', time.time() - TotalTime)


if __name__ == '__main__':
    GenerateLabelData()
