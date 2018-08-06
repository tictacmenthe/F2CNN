"""

 This file defines functions that allow to reorganise TIMIT and VTR_FORMANT files

Required structure:
root/
-scripts/   -> This file's directory
-resources/
--f2cnn/    -> The output directory
---TRAIN/
---TEST/
--TIMIT/    -> TIMIT database as is, including TEST and TRAIN directories
--VTR/      -> VTR FORMANTS files as is, including TEST and TRAIN subdirectories
-other directories/

"""
import time
from os import listdir, makedirs, mkdir
from os.path import isfile, splitext, dirname, exists, join, split, isdir
from shutil import copyfile

import glob

def completeSplit(filename):
    # Splitting filename
    splitted = []
    while True:
        file = split(filename)
        if file[0] in ('..', '.', '/'):
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


def moveFilesToPosition(vtrFileNames, timitFileNames):
    # Get the WAV files from timit
    source=[]
    dest=[]
    count=0
    for timitFile in timitFileNames:
        for vtrFile in vtrFileNames:
            if timitFile[-2].upper()==vtrFile[-2].upper() and timitFile[-1].upper()==vtrFile[-1].upper():
                src=join('resources', 'TIMIT', timitFile[-4], timitFile[-3], timitFile[-2], timitFile[-1])
                dst=join('resources', 'f2cnn', timitFile[-4].upper(), ".".join([timitFile[-3].upper(), timitFile[-2].upper(), timitFile[-1].upper()]))

                print("ENTRY:\t\t", src)
                print("\tCopying WAV file...")
                print("FROM\t\t", src+".WAV")
                print("TO\t\t\t", dst+".WAV")
                copyfile(src+".WAV", dst+".WAV")
                count+=1

                print("\tCopying PHN file...")
                print("FROM\t\t", src + ".PHN")
                print("TO\t\t\t", dst + ".PHN")
                copyfile(src+".PHN", dst+".PHN")
                count+=1

                print("\tCopying WRD file...")
                print("FROM\t\t", src + ".WRD")
                print("TO\t\t\t", dst + ".WRD")
                copyfile(src+".WRD", dst+".WRD")
                count+=1
                fbsrc=join('resources', 'VTR', vtrFile[-4], vtrFile[-3], vtrFile[-2], vtrFile[-1])
                print("\tCopying FB file...")
                print("FROM\t\t", fbsrc + ".fb")
                print("TO\t\t\t", dst + ".FB")
                copyfile(fbsrc+".fb", dst+".FB")
                count+=1



    print(count)



def OrganiseAllFiles(_):
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
