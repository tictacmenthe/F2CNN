"""

This file includes funcitons allowing to read phonemes from TIMIT database's .PHN files
"""

import csv

STOPS = ['b', 'd', 'g', 'p', 't', 'k', 'dx', 'q']
AFFRICATIVES = ['jh', 'ch']
FRICATIVES = ['s', 'sh', 'w', 'wh', 'f', 'th', 'v', 'dh']
NASALS = ['m', 'n', 'ng', 'em', 'en', 'eng', 'nx']
SEMIVOWELS_AND_GLIDES = ['l', 'r', 'w', 'y', 'hh', 'hv', 'el']
VOWELS = ["iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao",
          "oy", "ow", "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h"]

SILENTS = ['pau', 'epi', 'h#']


def ExtractPhonemes(phnFilename):
    data = []
    with open(phnFilename, 'r') as phnFile:
        reader = csv.reader(phnFile, delimiter=' ')
        for line in reader:
            data.append([int(line[0]), int(line[1]), line[2]])
    return data


def GetPhonemeAt(phnFilename, timepoint):
    """
    Returns the current phoneme at a TIMIT .WAV file frame
    :param phnFilename: the path to the .WAV file
    :param timepoint: the frame in the TIMIT .WAV file
    :return: the actual phoneme
    """
    data = ExtractPhonemes(phnFilename)
    for line in data:
        if line[0] <= timepoint <= line[1]:
            return line[2]
