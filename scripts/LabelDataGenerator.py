import glob
import time
import wave
from os.path import join
import numpy





def main():
    # # In case you need to print numpy outputs:
    # numpy.set_printoptions(threshold=numpy.inf, suppress=True)
    STEP = 160
    START = 5 * STEP
    TotalTime = time.time()

    # Get all the files under ../resources
    filenames = glob.glob(join("..", "resources", "f2cnn", "*", "*.WAV"))
    filenames = sorted(filenames)

    # # Test files
    # filenames = glob.glob(join("..", "testFiles", "*.WAV"))

    if not filenames:
        print("NO FILES FOUND")
        exit(-1)

    print(filenames)


    csvLines=[]

    # Get the data into a list of lists for the CSV
    for i, file in enumerate(filenames):
        print(i,file)
        wavFile=wave.open(file,'r')
        nb = int((wavFile.getnframes() / wavFile.getframerate() - 0.1) / 0.01)
        print(wavFile.getnframes(), nb)

        # Discretisation of the values for each entry required
        currentStep = START
        steps=[]
        for t in range(nb):
            entry = [currentStep + (k - 5) * STEP for k in range(11)]
            currentStep += STEP
            steps.append(entry)
        for step in steps:
            pass


    # Saving into a file
    with open("../trainingData/label_data.csv","w") as outputFile:
        pass

    print('                Total time:', time.time() - TotalTime)


if __name__ == '__main__':
    main()
