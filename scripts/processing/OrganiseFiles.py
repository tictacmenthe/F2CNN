"""
 This file defines functions that allow to reorganise TIMIT and VTR_FORMANT files
"""
import glob
import time
from os import mkdir
from os.path import splitext, join, split, isdir
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
        file = split(filename)
        if file[0] in ('..', '.', '/'):     # If we are at the end of a filepath:parent dir, current dir, root dir
            splitted.append(file[1])
            splitted.append(file[0])
            break
        elif file[0] == '':                 # If we were already at the end of the filepath
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
    count=0
    for timitFile in timitFileNames:
        for vtrFile in vtrFileNames:
            if timitFile[-2].upper()==vtrFile[-2].upper() and timitFile[-1].upper()==vtrFile[-1].upper():
                src=join('resources', 'TIMIT', timitFile[-4], timitFile[-3], timitFile[-2], timitFile[-1])
                dst=join('resources', 'f2cnn', timitFile[-4].upper(), ".".join([timitFile[-3].upper(), timitFile[-2].upper(), timitFile[-1].upper()]))

                print("ENTRY:\t", src)
                print("\t Copying WAV file...")
                print("FROM\t", src+".WAV")
                print("TO\t", dst+".WAV")
                copyfile(src+".WAV", dst+".WAV")
                count+=1

                print("\t Copying PHN file...")
                print("FROM\t", src + ".PHN")
                print("TO\t", dst + ".PHN")
                copyfile(src+".PHN", dst+".PHN")
                count+=1

                print("\t Copying WRD file...")
                print("FROM\t", src + ".WRD")
                print("TO\t", dst + ".WRD")
                copyfile(src+".WRD", dst+".WRD")
                count+=1
                fbsrc=join('resources', 'VTR', vtrFile[-4], vtrFile[-3], vtrFile[-2], vtrFile[-1])
                print("\t Copying FB file...")
                print("FROM\t", fbsrc + ".fb")
                print("TO\t", dst + ".FB\n")
                copyfile(fbsrc+".fb", dst+".FB")
                count+=1
    print(count, "files reorganized")


def OrganiseAllFiles():
    print("\n###############################\nReorganising files, like explained in the OrganiseFiles.py file's documentation.")
    TotalTime = time.time()
    vtrFileNames = list(map(lambda s:completeSplit(splitext(s)[0])[2:],glob.glob(join("resources","**","*.fb"),recursive=True)))
    timitFileNames = list(map(lambda s:completeSplit(splitext(s)[0])[2:],glob.glob(join("resources","**","*.WAV"), recursive=True)))
    if not vtrFileNames:
        vtrFileNames = list(map(lambda s: completeSplit(splitext(s)[0])[2:],
                                glob.glob(join("resources", "**", "*.FB"), recursive=True)))
        if not vtrFileNames:
            print("NO VTR FILES FOUND")
            exit(-1)
    if not timitFileNames:
        print("NO TIMIT FILES FOUND")
        exit(-1)

    print("FOUND",len(vtrFileNames), "VTR FB FILES TO ORGANIZE.")
    print("FOUND",len(timitFileNames), "TIMIT WAV FILES.")
    if not isdir(join('resources')):
        mkdir('resources')

    if not isdir(join('resources', 'f2cnn')):
        mkdir(join('resources', 'f2cnn'))
        mkdir(join('resources', 'f2cnn', 'TEST'))
        mkdir(join('resources', 'f2cnn', 'TRAIN'))

    if not isdir(join('resources', 'f2cnn', 'TRAIN')):
        mkdir(join('resources', 'f2cnn', 'TRAIN'))
    if not isdir(join('resources', 'f2cnn', 'TEST')):
        mkdir(join('resources', 'f2cnn', 'TEST'))

    moveFilesToPosition(vtrFileNames, timitFileNames)

    print("Done reorganizing files.")
    print('                Total time:', time.time() - TotalTime)
    print('')
