import glob
import time
from os.path import join
import numpy


def main():
    # # In case you need to print numpy outputs:
    # numpy.set_printoptions(threshold=numpy.inf, suppress=True)
    SAMPLING_FREQ = 16000.
    STEP = 160
    START = 5 * STEP
    TotalTime = time.time()

    # Get all the ENV1.npy files under ../resources
    envFiles = glob.glob(join("..", "resources", "f2cnn", "*", "*.ENV1.npy"))
    envFiles = sorted(envFiles)

    # # Test files
    # envFiles = glob.glob(join("..", "testFiles","*.ENV1.npy"))

    if not envFiles:
        print("NO ENV1.npy FILES FOUND")
        exit(-1)

    print(envFiles)

    inputData = []
    for i, file in enumerate(envFiles):
        print(i,file)
        envelopes = numpy.load(file)

        # Number of timepoints for measures
        nb = int((len(envelopes[0]) / SAMPLING_FREQ - 0.11) / 0.01)

        # Discretisation of the values for each entry required
        currentStep = START
        steps=[]
        for t in range(nb):
            entry = [currentStep + (k - 5) * STEP for k in range(11)]
            currentStep += STEP
            steps.append(entry)

        for entry in steps:
            entryMatrix=[]  # All the values for one entry
            for index in entry:
                valueArray=[channel[index] for channel in envelopes] # All the values of env at the steps' timepoint
                entryMatrix.append(valueArray)
            inputData.append(entryMatrix)
        print(file, "done")
    sizes=(len(inputData),len(inputData[0]),len(inputData[0][0]))
    print(sizes)
    inputData=numpy.array(inputData, dtype=numpy.float32)
    print(inputData.shape)
    numpy.save("../trainingData/input_data",inputData)
    print('                Total time:', time.time() - TotalTime)


if __name__ == '__main__':
    main()
