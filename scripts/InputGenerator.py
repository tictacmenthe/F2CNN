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

    trainingData = []
    for i, file in enumerate(envFiles):
        print(i,file)
        envelopes = numpy.load(file)
        nb = int((len(envelopes[0]) / SAMPLING_FREQ - 0.1) / 0.01)
        print(len(envelopes[0]), nb)

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
            trainingData.append(entryMatrix)
        print(file, "done")
    sizes=(len(trainingData),len(trainingData[0]),len(trainingData[0][0]))
    print(sizes)
    print(trainingData)
    trainingData=numpy.array(trainingData, dtype=numpy.float32)
    print(trainingData.shape)
    numpy.save("../inputData/training_data",trainingData)
    print('                Total time:', time.time() - TotalTime)


if __name__ == '__main__':
    main()
