"""

This file allows the reading of VTR Formant database's .FB files, containing the F1,F2,F3 and F3 formants' frequencies
and bandwiths.
The .FB files should be organised like the output of the OrganiseFiles.py script,
inside ../f2cnn/TEST OR TRAIN/ with the names DRr.reader.sentence.FB, with r the regionm reader the ID of the reader
and sentence the ID of the sentence read.
The output is a numpy.ndarray of size 8*nb_frames, with bn_frames being one of the header parameters of the .FB file.

"""

import csv
import glob
import time
from os.path import join

import numpy


def ExtractPhonemes(phnFilename):
    data = []
    with open(phnFilename, 'r') as phnFile:
        reader = csv.reader(phnFile, delimiter=' ')
        for line in reader:
            data.append([int(line[0]), int(line[1]), line[2]])
    return data


def GetPhonemeAt(phnFilename, timepoint):
    data = ExtractPhonemes(phnFilename)
    for line in data:
        if line[0] <= timepoint <= line[1]:
            return line[2]


def main():
    # # In case you need to print numpy outputs:
    numpy.set_printoptions(threshold=numpy.inf, suppress=True)
    print("Extraction of FB Files...")
    TotalTime = time.time()

    # # Get all the PHN files under ../resources
    # phnFiles = glob.glob(join("..","resources","f2cnn","*","*.PHN"))

    # Test PHN files
    phnFiles = glob.glob(join("..", "testFiles", "*.PHN"))

    if not phnFiles:
        print("NO FB FILES FOUND")
        exit(-1)
    print(phnFiles)

    ph = GetPhonemeAt(phnFiles[0], 19000)
    print(ph)

    print('              Total time:', time.time() - TotalTime)


if __name__ == '__main__':
    main()
