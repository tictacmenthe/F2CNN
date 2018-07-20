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

from os import listdir, makedirs
from os.path import isfile, splitext, dirname, exists, join, split
from shutil import copyfile

DIRSRC = "resources"
DIRVTR = join(DIRSRC, "vtr_formants")
DIRTIM = join(DIRSRC, "TIMIT")
DIROUTPUT = join(DIRSRC, "f2cnn")

# 0=TIMIT, 1=VTR
REGNUMTEST = [1, 2, 3, 4, 5, 6, 7, 8]
REGNUMTRAIN = [1, 3, 4, 5, 6, 7]
REGNUM = [REGNUMTRAIN, REGNUMTEST]
CASE = ['TRAIN', 'TEST']


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
                    filepath = join(p_path,f)
                    if isfile(filepath):
                        output.append(filepath)
    return output


def splitVTRFileNames(files):
    splittedFileNames = []
    for name in files:
        splittedFileNames.append(split(name.upper())[3:])
    return splittedFileNames


def moveFilesToPosition(files):
    # Get the WAV files from timit
    upperFiles = []
    for f in files:
        print(split(splitext(f)[0]))
        inTimit = join(split(splitext(f)[0])[3:]).upper()
        print("INTIMIT",inTimit)
        upperFiles.append(inTimit)
    upperFiles = set(upperFiles)
    for i,f in enumerate(upperFiles):
        path = join(DIRTIM,f + '.WAV')
        f = split(f)
        # The output filename is REGION.PERSON.SENTENCE.EXTENSION
        f = join(f[0],f[1] + '.' + f[2] + '.' + f[3])
        newPath = join(DIROUTPUT,f + '.WAV')
        if isfile(path):
            if not exists(dirname(newPath)):
                makedirs(dirname(newPath))
            print(i,'Copying',path,'to\n',newPath)
            copyfile(path, newPath)
        else:
            print("DOESNT EXIST",path)
    # Get the other files
    splittedFileNames = splitVTRFileNames(files)
    for src, dst in zip(files, splittedFileNames):
        dst = join(DIROUTPUT,dst[0],dst[1] + '.' + dst[2] + '.' + dst[3])
        print('Copying', src, 'to\n', dst)
        copyfile(src, dst)


def main():
    fileNames = getVTRFileNames()
    moveFilesToPosition(fileNames)


if __name__ == '__main__':
    main()
