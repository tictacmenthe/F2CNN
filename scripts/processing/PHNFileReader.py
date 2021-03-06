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
    try:
        with open(phnFilename, 'r') as phnFile:
            reader = csv.reader(phnFile, delimiter=' ')
            for line in reader:
                data.append((line[2], int(line[0]), int(line[1])))  # (phoneme, start, end) tuples
        return data
    except FileNotFoundError:
        print("No .PHN phoneme data file.")
        return None


def GetPhonemeFromArrayAt(phonemes, timepoint):
    for line in phonemes:
        if line[1]<=timepoint<=line[2]:
            return line[0]
    return 'h#'


def GetPhonemeAt(phnFilename, timepoint):
    """
    Returns the current phoneme at a TIMIT .WAV file frame
    :param phnFilename: the path to the .WAV file
    :param timepoint: the frame in the TIMIT .WAV file
    :return: the actual phoneme
    """
    return GetPhonemeFromArrayAt(ExtractPhonemes(phnFilename), timepoint)
