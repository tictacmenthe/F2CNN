from __future__ import print_function

import csv
import typing
import keras
import numpy
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from matplotlib import pyplot
import numpy
from matplotlib.colors import LogNorm

from gammatone.filters import centre_freqs
from scripts.EnvelopeExtraction import ExtractEnvelopeFromMatrix
from scripts.GammatoneFiltering import GetFilteredOutputFromFile
from scripts.Plotting import GetNewHeightERB

numpy.set_printoptions(threshold=numpy.inf)


class BatchPlotter(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.acc = []
        self.loss = []

    def on_batch_end(self, batch, logs=None):
        self.acc.append(logs['acc'])
        self.loss.append(logs['loss'])


def normalizeInput(matrix: numpy.ndarray):
    minvalue, maxvalue = matrix.min(), matrix.max()
    if minvalue > maxvalue:
        raise ValueError("minvalue must be less than or equal to maxvalue")
    elif minvalue <= 0:
        raise ValueError("values must all be positive")
    elif minvalue == maxvalue:
        matrix.fill(0)
        return matrix
    minvalue, maxvalue = numpy.log(matrix.min()), numpy.log(matrix.max())

    logMatrix = numpy.log(matrix)
    logMatrix -= minvalue
    logMatrix /= maxvalue - minvalue
    return logMatrix


def SeparateTestTrain(pathToInput, pathToLabel):
    x = [[], []]
    y = [[], []]
    currentCase = 0
    lastRegion = 'DR1'
    input_data = numpy.load(pathToInput)
    with open(pathToLabel, 'r') as labels:
        reader = csv.reader(labels)
        for i, (region, speaker, sentence, phoneme, timepoint, slope, pvalue, sign) in enumerate(reader):
            if region != lastRegion:  # If there is a change in region
                if region == 'DR1':  # And the change is to a DR1
                    currentCase += 1  # It means we transition form TEST to TRAIN
                lastRegion = region
            x[currentCase].append(input_data[i])
            y[currentCase].append(int(sign))
    return numpy.array(x[0]), numpy.array(y[0]), numpy.array(x[1]), numpy.array(y[1])


def TrainAndPlotLoss():
    batch_size = 128
    num_classes = 2
    epochs = 20
    # input image dimensions
    img_rows, img_cols = 11, 128

    x_test, y_test, x_train, y_train = SeparateTestTrain('trainingData/input_data.npy',
                                                         'trainingData/label_data.csv')

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # TODO: normalize
    for i, matrix in enumerate(x_train):
        x_train[i] = normalizeInput(matrix)
    for i, matrix in enumerate(x_test):
        x_test[i] = normalizeInput(matrix)

    print('Rising test:', len([sign for sign in y_test if sign == 1]))
    print('Falling test:', len([sign for sign in y_test if sign == 0]))
    print('Rising train:', len([sign for sign in y_train if sign == 1]))
    print('Falling train:', len([sign for sign in y_train if sign == 0]))

    print(x_train.shape, 'train samples')
    print(x_test.shape, 'test samples')
    print(y_test[:10])

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print("Categories: [falling, rising]")
    print(y_test[:10])

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    stopCallback = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.01, patience=5, verbose=1, mode='auto',
                                                 baseline=None)

    batchPlotCallback = BatchPlotter()

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[batchPlotCallback, stopCallback],
                        verbose=1,
                        validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=1)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    fig = pyplot.figure()
    val_acc = fig.add_subplot(121)
    acc = fig.add_subplot(122)

    acc.plot(batchPlotCallback.acc, label='Traing Accuracy')
    val_acc.plot(history.history['val_acc'], label='Validation Accuracy')
    acc.set_xlabel("Batch Number")
    val_acc.set_xlabel("Epoch")
    acc.set_ylabel("Accuracy")
    val_acc.set_ylabel("Validation Accuracy")
    pyplot.legend()
    pyplot.show(fig)
    model.save('trained_model')
    model.save_weights('trained_model_weights')
    with open('trained_model_json', 'w') as jsonfile:
        jsonfile.write(model.to_json())
    model = keras.models.load_model('trained_model')
    model.evaluate(x_test, y_test)
    print(model.predict(numpy.array([x_test[0]])))
    print(model.predict(numpy.array([x_test[1]])))
    print(model.predict(numpy.array([x_test[2]])))
    print(model.predict(numpy.array([x_test[3]])))
    print(y_test[:4])


def EvaluateOneFile(wavFileName='testFiles/DR3.FCKE0.SI1111.WAV'):
    filtered = GetFilteredOutputFromFile(wavFileName)
    envelope = ExtractEnvelopeFromMatrix(filtered)
    nb = int(((len(envelope[0]) / 16000) - 0.11) * 16000)
    input_data = numpy.zeros([nb, 11, 128])
    print(input_data.shape, envelope.shape)
    START = int(0.055 * 16000)
    STEP = 160
    for i in range(0, nb):
        input_data[i] = [[channel[START + i + (k - 5) * STEP] for channel in envelope] for k in range(11)]

    for i, matrix in enumerate(input_data):
        input_data[i] = normalizeInput(matrix)

    print(input_data.shape)
    input_data.astype('float32')
    model = keras.models.load_model('trained_model')

    scores = model.predict(input_data.reshape(nb, 11, 128, 1))
    print(scores.shape)
    CENTER_FREQUENCIES = centre_freqs(16000, 128, 100)
    h, ratios = GetNewHeightERB(envelope, CENTER_FREQUENCIES)
    image = numpy.zeros([h, envelope.shape[1]])
    i = 0
    r = 0
    for line in envelope:
        j = 0
        for j in range(ratios[r]):
            image[i + j] = line
        i += j + 1
        r += 1

    scoreheight=[1 if score[1]>score[0] else 0 for score in scores]

    print(len(scoreheight))
    # pyplot.imshow(image, norm=LogNorm(), aspect="auto", extent=[0, len(envelope[0]) / 16000., 100, 7795])
    pyplot.bar([i for i in range(len(scoreheight))], scoreheight)
    pyplot.show()


if __name__ == '__main__':
    TrainAndPlotLoss()
