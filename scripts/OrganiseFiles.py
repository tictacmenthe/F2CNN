"""

 This file defines functions that allow to reorganise TIMIT and VTR_FORMANT files

Required structure:
root/
-scripts/ -> This file's directory
-resources/
--f2cnn/ -> The output directory
---TRAIN/
---TEST/
--timit/
---TIMIT/ -> TIMIT database as is, including TEST and TRAIN directories
--vtr_formants/ -> VTR FORMANTS files as is, including TEST and TRAIN subdirectories
-other directories/

"""
import time
from os import listdir, makedirs
from os.path import isfile, splitext, dirname, exists, join, split
from shutil import copyfile

DIRSRC = "resources"
DIRVTR = join(DIRSRC, "vtr_formants")
DIRTIM = join(DIRSRC, "TIMIT")
DIROUTPUT = join(DIRSRC, "f2cnn")
# Regions
REGNUMTEST = [1, 2, 3, 4, 5, 6, 7, 8]
REGNUMTRAIN = [1, 3, 4, 5, 6, 7]
REGNUM = [REGNUMTRAIN, REGNUMTEST]
CASE = ['TRAIN', 'TEST']


def completeSplit(filename):
    # Splitting filename
    splitted = []
    while True:
        file = split(filename)
        if file[0] in ('..', '.'):
            splitted.append(file[1])
            splitted.append(file[0])
            break
        elif file[0] == '':
            splitted.append(file[1])
            break
        splitted.append(file[1])
        filename = file[0]
    splitted.reverse()
    return splitted


def getVTRFileNames():
    output = []
    for i, c in enumerate(CASE):
        for r in REGNUM[i]:
            reg_path = join(DIRVTR, c, 'dr' + str(r))
            people = listdir(reg_path)
            for p in people:
                p_path = join(reg_path, p)
                files = listdir(p_path)
                for f in files:
                    filepath = join(p_path, f)
                    if isfile(filepath):
                        output.append(filepath)
    return output


def splitVTRFileNames(files):
    splittedFileNames = []
    for name in files:
        splittedFileNames.append(completeSplit(name.upper())[2:])
    return splittedFileNames


def moveFilesToPosition(files):
    # Get the WAV files from timit
    upperFiles = []
    for f in files:
        # Remove extension
        f = splitext(f)[0]
        # Split
        splitted = completeSplit(f)
        inTimit = join(*(splitted[-4:])).upper()
        print(inTimit)
        upperFiles.append(inTimit)
    upperFiles = set(upperFiles)
    for i, f in enumerate(upperFiles):
        path = join(DIRTIM, f + '.WAV')
        f = completeSplit(f)
        # The output filename is REGION.PERSON.SENTENCE.EXTENSION
        f = join(f[0], f[1] + '.' + f[2] + '.' + f[3])
        newPath = join(DIROUTPUT, f + '.WAV')
        if isfile(path):
            if not exists(dirname(newPath)):
                makedirs(dirname(newPath))
            print('Copying\t\t{}\nto\t\t{}'.format(path, newPath))
            copyfile(path, newPath)
        else:
            print("DOESNT EXIST", path)
    # Get the other files
    splittedFileNames = splitVTRFileNames(files)
    print(splittedFileNames)
    for src, dst in zip(files, splittedFileNames):
        dst = join(DIROUTPUT, dst[0], dst[1] + '.' + dst[2] + '.' + dst[3])
        print('Copying\t\t{}\nto\t\t{}'.format(src, dst))
        copyfile(src, dst)


def OrganiseAllFiles(_):
    print("\n###############################\nReorganising files, like in the OrganiseFiles.py file's documentation.")
    TotalTime = time.time()
    fileNames = getVTRFileNames()
    moveFilesToPosition(fileNames)
    print("Done reorganizing files.")
    print('                Total time:', time.time() - TotalTime)
    print('')
