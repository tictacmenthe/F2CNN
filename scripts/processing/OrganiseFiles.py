"""
 This file defines functions that allow to reorganise TIMIT and VTR_FORMANT files
"""
import glob
import os
import time
from shutil import copyfile


def completeSplit(filename):
    """
    Splits the path to a file
    :param filename: path to the file
    :return:    list of splitted filepath
    """
    # Splitting filename
    splitted = []
    while True:
        file = os.path.split(filename)
        if file[0] in ('..', '.', '/'):  # If we are at the end of a filepath:parent dir, current dir, root dir
            splitted.append(file[1])
            splitted.append(file[0])
            break
        elif file[0] == '':  # If we were already at the end of the filepath
            splitted.append(file[1])
            break
        splitted.append(file[1])
        filename = file[0]
    splitted.reverse()
    return splitted


def moveFilesToPosition(vtrFileNames, timitFileNames):
    """
    Moves all files to their correct position in ./resources/f2cnn
    :param vtrFileNames: paths to all vtr .FB files
    :param timitFileNames: paths to all Timit .WAV files
    """
    # Get the WAV files from timit
    count = 0
    notfound = 0
    for timitFile in timitFileNames:
        for vtrFile in vtrFileNames:
            if timitFile[-2].upper() == vtrFile[-2].upper() and timitFile[-1].upper() == vtrFile[-1].upper():
                src = os.path.join('resources', 'TIMIT', timitFile[-4], timitFile[-3], timitFile[-2], timitFile[-1])
                dst = os.path.join('resources', 'f2cnn', timitFile[-4].upper(),
                                   ".".join([timitFile[-3].upper(), timitFile[-2].upper(), timitFile[-1].upper()]))

                print("ENTRY:\t", src)
                try:
                    copyfile(src + ".WAV", dst + ".WAV")
                    count += 1
                except FileNotFoundError:
                    print('ERROR: FILENOTFOUND DURING')
                    print("\t Copying WAV file")
                    print("FROM\t", src + ".WAV")
                    print("TO\t", dst + ".WAV")
                    notfound += 1

                try:
                    copyfile(src + ".PHN", dst + ".PHN")
                    count += 1
                except FileNotFoundError:
                    print('ERROR: FILENOTFOUND DURING')
                    print("\t Copying PHN file")
                    print("FROM\t", src + ".PHN")
                    print("TO\t", dst + ".PHN")
                    notfound += 1

                try:
                    copyfile(src + ".WRD", dst + ".WRD")
                    count += 1
                except FileNotFoundError:
                    print('ERROR: FILENOTFOUND DURING')
                    print("\t Copying WRD file")
                    print("FROM\t", src + ".WRD")
                    print("TO\t", dst + ".WRD")
                    notfound += 1

                fbsrc = os.path.join('resources', 'VTR', vtrFile[-4], vtrFile[-3], vtrFile[-2], vtrFile[-1])
                try:
                    copyfile(fbsrc + ".fb", dst + ".FB")
                    count += 1
                except FileNotFoundError:
                    print('ERROR: FILENOTFOUND DURING')
                    print("\t Copying FB file")
                    print("FROM\t", fbsrc + ".fb")
                    print("TO\t", dst + ".FB\n")
                    notfound += 1

    print(count, "files reorganized.")
    if notfound > 0:
        print(notfound, "files not found.")


def OrganiseAllFiles():
    print(
        "\n###############################\nReorganising files, like explained in the OrganiseFiles.py file's documentation.")
    TotalTime = time.time()
    vtrFileNames = list(
        map(lambda s: completeSplit(os.path.splitext(s)[0])[2:],
            glob.glob(os.path.join("resources", "**", "*.fb"), recursive=True)))
    timitFileNames = list(
        map(lambda s: completeSplit(os.path.splitext(s)[0])[2:],
            glob.glob(os.path.join("resources", "**", "*.WAV"), recursive=True)))
    if not vtrFileNames:
        vtrFileNames = list(map(lambda s: completeSplit(os.path.splitext(s)[0])[2:],
                                glob.glob(os.path.join("resources", "**", "*.FB"), recursive=True)))
        if not vtrFileNames:
            print("NO VTR FILES FOUND")
            exit(-1)
    if not timitFileNames:
        print("NO TIMIT FILES FOUND")
        exit(-1)

    print("FOUND", len(vtrFileNames), "VTR FB FILES TO ORGANIZE.")
    print("FOUND", len(timitFileNames), "TIMIT WAV FILES.")
    TEST_DIR = os.path.join('resources', 'f2cnn', 'TEST')
    TRAIN_DIR = os.path.join('resources', 'f2cnn', 'TRAIN')
    os.makedirs(os.path.split(TEST_DIR)[0], exist_ok=True)
    os.makedirs(os.path.split(TRAIN_DIR)[0], exist_ok=True)

    moveFilesToPosition(vtrFileNames, timitFileNames)

    print("Done reorganizing files.")
    print('                Total time:', time.time() - TotalTime)
    print('')
