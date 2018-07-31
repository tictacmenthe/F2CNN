"""

This file generates labelling data for the CNN, as a .CSV file of columns:
Region(DR1-8),SpeakerID,SentenceID,framepoint,slope,p-valueOfSlope,slopeSign(+-1)
Requires a prior execution of the OrganiseFiles.py, GammatoneFiltering.py, EnvelopeExtraction.py scripts' main functions

"""
from configparser import ConfigParser

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
        filepath = 'testFiles/trainingData/label_data.csv'
    else:
        filepath = 'trainingData/label_data.csv'
    with open(filepath, 'r') as labelDataFile:
        csvReader = csv.reader(labelDataFile)
        return [int(line[-1]) for line in csvReader]


def GenerateLabelData(testMode):
    TotalTime = time.time()

    # #### READING CONFIG FILE
    config = ConfigParser()
    config.read('F2CNN.conf')
    radius = config.getint('CNN', 'RADIUS')
    RISK = config.getfloat('CNN', 'RISK')
    ustos=1./1000000
    dotsperinput=radius*2+1

    if testMode:
        # Test files
        filenames = glob.glob(join("testFiles", "*.WAV"))
    else:
        # Get all the files under resources
        filenames = glob.glob(join("resources", "f2cnn", "*", "*.WAV"))

    print("\n###############################\nGenerating Label Data from files in '{}'.".format(
        split(split(filenames[0])[0])[0]))

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
        # Load the F2 values of the file
        F2Array, sampPeriod = GetF2Frequencies(splitext(file)[0] + '.FB')

        # Get number of points
        wavFile = wave.open(file, 'r')
        framerate=wavFile.getframerate()
        nb = int((wavFile.getnframes() / framerate - dotsperinput*sampPeriod*ustos)*100)
        wavFile.close()

        # Get the information about the person
        region, speaker, sentence, _ = split(file)[1].split(".")


        # Discretization of the values for each entry required
        STEP = int(framerate*sampPeriod*ustos)
        START = int(STEP * (radius + 0.5))  # add a 0.5 dot margin, to prevent marginal cases
        currentStep = START
        steps = []
        for t in range(nb):
            entry = [currentStep + (k - radius) * STEP for k in range(dotsperinput)]
            currentStep += STEP
            steps.append(entry)

        for step in steps:
            phoneme = GetPhonemeAt(splitext(file)[0] + '.PHN', step[radius])
            entry = [region, speaker, sentence, phoneme, step[radius]]
            F2Values = numpy.array(GetF2FrequenciesAround(F2Array, step[radius], radius))

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
                entry.append(1 if a > 0 else 0)
                csvLines.append(entry)
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
