"""

This file generates labelling data for the CNN, as a .CSV file of columns:
TESTorTRAIN,Region(DR1-8),SpeakerID,SentenceID,framepoint,slope,p-valueOfSlope,slopeSign(+-1)
Requires a prior execution of the OrganiseFiles.py, GammatoneFiltering.py, EnvelopeExtraction.py scripts' main functions

"""
import csv
import glob
import time
from configparser import ConfigParser
from os.path import join, split, splitext

import numpy
from scipy.stats import pearsonr

from scripts.processing.GammatoneFiltering import GetArrayFromWAV
from .FBFileReader import GetFromantFrequenciesAround, GetFormantFrequencies
from .PHNFileReader import ExtractPhonemes, SILENTS, GetPhonemeFromArrayAt


def ExtractLabel(wavFile, config):
    fileBase=splitext(wavFile)[0]
    # #### READING CONFIG FILE
    RADIUS = config.getint('CNN', 'RADIUS')
    CENTERED = config.getboolean('CNN', 'CENTERED')
    RISK = config.getfloat('CNN', 'RISK')
    FORMANT = config.getint('CNN', 'FORMANT')
    SAMPPERIOD = config.getint('CNN', 'SAMPPERIOD')
    DOTSPERINPUT = RADIUS * 2 + (1 if CENTERED else 0)
    USTOS = 1.0 / 1000000

    # Load the F2 values of the file
    FormantArray, _ = GetFormantFrequencies(fileBase + '.FB', FORMANT)
    if FormantArray is None:
        return None
    phonemes = ExtractPhonemes(fileBase + '.PHN')
    print(phonemes)
    exit()
    # Get number of points
    framerate, wavList = GetArrayFromWAV(wavFile)
    wavToFormant = framerate * SAMPPERIOD * USTOS
    nf = len(wavList)
    nb = int(nf / (framerate * SAMPPERIOD * USTOS) - DOTSPERINPUT - 1)

    # Get the information about the person
    region, speaker, sentence = split(fileBase)[1].split(".")
    testOrTrain = split(split(fileBase)[0])[1]

    # Discretization of the values for each entry required
    STEP = int(framerate * SAMPPERIOD * USTOS)
    START = int(STEP * RADIUS)
    steps = [START + k * STEP for k in range(nb)]

    output=[]
    # Getting the data for each 'step'
    for step in steps:
        phoneme = GetPhonemeFromArrayAt(phonemes, step)  # Extraction of the required phonemes
        if phoneme in SILENTS:  # Silent category of phonemes will be ignored
            continue
        if not phoneme:  # End case: there is no phoneme to be read
            break
        entry = [testOrTrain, region, speaker, sentence, phoneme, step]
        FormantValues = numpy.array(GetFromantFrequenciesAround(FormantArray, step, RADIUS, wavToFormant))

        # Least Squares Method for linear regression of the F2 values
        x = numpy.array([step + (k - 5) * STEP for k in range(DOTSPERINPUT)])
        A = numpy.vstack([x, numpy.ones(len(x))]).T
        [a, b], _, _, _ = numpy.linalg.lstsq(A, FormantValues, rcond=None)
        # Pearson Correlation Coefficient r and p-value p using scipy.stats.pearsonr
        r, p = pearsonr(FormantValues, a * x + b)
        # We round them up at 5 digits after the comma
        entry.append(round(a, 5))
        entry.append(round(p, 5))
        # The line to be added to the CSV file, only if the direction of the formant is clear enough (% risk)

        if p < RISK:
            entry.append(1 if a > 0 else 0)
            output.append(entry)
    return output if len(output)>0 else None


def GenerateLabelData():
    TotalTime = time.time()

    # #### READING CONFIG FILE
    config = ConfigParser()
    config.read('F2CNN.conf')

    # Get all the files under resources
    filenames = glob.glob(join("resources", "f2cnn", "*", "*.WAV"))
    print("\n###############################\nGenerating Label Data from files in '{}' into 2 classes.".format(
        split(split(filenames[0])[0])[0]))

    # Alphanumeric order
    filenames = sorted(filenames)

    if not filenames:
        print("NO FILES FOUND")
        exit(-1)

    nfiles = len(filenames)
    print(nfiles, "files found")

    csvLines = []

    # Get the data into a list of lists for the CSV
    for i, file in enumerate(filenames):
        print("Reading:\t{:<50}\t{}/{}".format(file, i, nfiles))
        fileEntry = ExtractLabel(file, config)
        if fileEntry is not None:
            csvLines.extend(fileEntry)
        print("\t\t{:<50}\tdone !".format(file))

    # Saving into a file
    filePath = join("trainingData", "label_data.csv")
    with open(filePath, "w") as outputFile:
        writer = csv.writer(outputFile)
        for line in csvLines:
            writer.writerow(line)
    print("Generated Label Data CSV of", len(csvLines), "lines.")
    print('                Total time:', time.time() - TotalTime)
    print('')
