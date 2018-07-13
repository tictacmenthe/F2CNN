import glob
import time
import wave
from os.path import join, split, basename, splitext
import numpy
import csv
import matplotlib.pyplot as plt
from scripts.PHNFileReader import GetPhonemeAt
from scripts.FBFileReader import GetF2Frequencies,GetF2FrequenciesAround

def main():
    # # In case you need to print numpy outputs:
    # numpy.set_printoptions(threshold=numpy.inf, suppress=True)
    STEP = 160
    START = 5 * STEP
    TotalTime = time.time()

    # # Get all the files under ../resources
    # filenames = glob.glob(join("..", "resources", "f2cnn", "*", "*.WAV"))

    # Test files
    filenames = glob.glob(join("..", "testFiles", "*.WAV"))
    filenames = sorted(filenames)

    if not filenames:
        print("NO FILES FOUND")
        exit(-1)

    print(filenames)

    csvLines = []

    # Get the data into a list of lists for the CSV
    for i, file in enumerate(filenames):
        print(i, file)

        # Get number of points
        wavFile = wave.open(file, 'r')
        nb = int((wavFile.getnframes() / wavFile.getframerate() - 0.1) / 0.01)
        print(wavFile.getnframes(), nb)
        wavFile.close()

        # Get the information about the person
        fileData = split(file)[1].split(".")
        region, speaker, sentence, _ = fileData

        # Discretisation of the values for each entry required
        currentStep = START
        steps = []
        for t in range(nb):
            entry = [currentStep + (k - 5) * STEP for k in range(11)]
            currentStep += STEP
            steps.append(entry)


        # Load the F2 values of the file
        F2Array,_=GetF2Frequencies(splitext(file)[0]+'.FB')
        for step in steps:
            entry=[region,speaker,sentence, GetPhonemeAt(splitext(file)[0]+'.PHN', step[5]), step[5]]
            F2Values=numpy.array(GetF2FrequenciesAround(F2Array,step[5],5))
            x=numpy.array([i for i in range(11)])
            A = numpy.vstack([x, numpy.ones(len(x))]).T
            [a, b],residuals,_, _  = numpy.linalg.lstsq(A,F2Values, rcond=None)
            print(residuals)
            output=a*x+b
            print(a,b)
            plt.plot(F2Values,'r.')
            plt.plot(x,output)
            plt.show()
            csvLines.append(entry)

    # Saving into a file
    with open("../trainingData/label_data.csv", "w") as outputFile:
        writer = csv.writer(outputFile)
        for line in csvLines:
            writer.writerow(line)

    print('                Total time:', time.time() - TotalTime)


if __name__ == '__main__':
    main()
