"""

 This file defines functions that allow to reorganise TIMIT and VTR_FORMANT files

Required structure:
root/
-scripts/ -> This file's directory
-src/
--f2cnn/ -> The output directory
---TRAIN/
---TEST/
--timit/
---TIMIT/ -> TIMIT database as is, including TEST and TRAIN directories
--vtr_formants/ -> VTR FORMANTS files as is, including TEST and TRAIN subdirectories
-other directories/

"""

from os import listdir, makedirs
from os.path import isfile, splitext, dirname, exists
from shutil import copyfile

SEP = '/'
DIRSRC = "../src/"
DIRVTR = DIRSRC + "vtr_formants"
DIRTIM = DIRSRC + "timit/TIMIT"
DIROUTPUT = DIRSRC + 'f2cnn'

# 0=TIMIT, 1=VTR
REGNUMTEST = [1, 2, 3, 4, 5, 6, 7, 8]
REGNUMTRAIN = [1, 3, 4, 5, 6, 7]
REGNUM = [REGNUMTRAIN, REGNUMTEST]
CASE = ['TRAIN', 'TEST']


def getVTRFileNames():
    output = []

    for i, c in enumerate(CASE):
        for r in REGNUM[i]:
            reg_path = DIRVTR + SEP + c + SEP + 'dr' + str(r)
            people = listdir(reg_path)
            for p in people:
                p_path = reg_path + SEP + p
                files = listdir(p_path)
                for f in files:
                    filepath = p_path + SEP + f
                    if isfile(filepath):
                        output.append(filepath)
    return output


def splitVTRFileNames(files):
    splittedFileNames = []
    for name in files:
        splittedFileNames.append(name.upper().split('/')[3:])
    return splittedFileNames


def moveFilesToPosition(files):
    # Get the WAV files from timit
    upperFiles = []
    for f in files:
        inTimit = '/'.join(splitext(f)[0].split('/')[3:]).upper()
        upperFiles.append(inTimit)
    upperFiles = set(upperFiles)
    for f in upperFiles:
        path = DIRTIM + SEP + f + '.WAV'
        f = f.split('/')
        f = f[0] + SEP + f[1] + '.' + f[2] + '.' + f[3]
        newPath = DIROUTPUT + SEP + f + '.WAV'
        if isfile(path):
            if not exists(dirname(newPath)):
                makedirs(dirname(newPath))
            # print('Copying',path,'to\n',newPath)
            copyfile(path, newPath)
    # Get the other files
    splittedFileNames = splitVTRFileNames(files)
    for src, dst in zip(files, splittedFileNames):
        dst = DIROUTPUT + SEP + dst[0] + SEP + dst[1] + '.' + dst[2] + '.' + dst[3]
        print('Copying', src, 'to\n', dst)
        copyfile(src, dst)


def main():
    fileNames = getVTRFileNames()
    moveFilesToPosition(fileNames)


if __name__ == '__main__':
    main()
