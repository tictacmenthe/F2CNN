from __future__ import print_function

import csv
import glob
import os
import time
from configparser import ConfigParser
import numpy


from matplotlib import pyplot

from gammatone import filters
from scripts.processing.EnvelopeExtraction import ExtractEnvelopeFromMatrix
from scripts.processing.FBFileReader import ExtractFBFile
from scripts.processing.GammatoneFiltering import GetFilteredOutputFromFile
from scripts.processing.PHNFileReader import ExtractPhonemes
from scripts.Plotting import PlotEnvelopesAndCNNResultsWithPhonemes

numpy.set_printoptions(threshold=numpy.inf)


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
    input_data = numpy.load(pathToInput)
    with open(pathToLabel, 'r') as labels:
        reader = csv.reader(labels)
        for i, (test, region, speaker, sentence, phoneme, timepoint, slope, pvalue, sign) in enumerate(reader):
            test=test=='TEST'
            x[0 if test else 1].append(input_data[i])
            y[0 if test else 1].append(int(sign))
    return numpy.array(x[0]), numpy.array(y[0]), numpy.array(x[1]), numpy.array(y[1])


def TrainAndPlotLoss():
    from keras.layers import Conv2D, MaxPooling2D, Activation
    from keras.layers import Dense, Dropout, Flatten
    from keras.models import Sequential
    from keras.utils import to_categorical
    from keras.optimizers import rmsprop
    from keras.losses import categorical_crossentropy
    from keras.callbacks import EarlyStopping

    batch_size = 32
    num_classes = 2
    epochs = 50
    # input image dimensions
    img_rows, img_cols = 11, 128

    x_test, y_test, x_train, y_train = SeparateTestTrain('trainingData/input_data.npy',
                                                         'trainingData/label_data.csv')

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

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

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    print("Categories: [falling, rising]")

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(516))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = rmsprop(lr=0.0001, decay=1e-6)

    model.compile(loss=categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])
    stopCallback = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=6, verbose=1, mode='auto',
                                                 baseline=None)

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[stopCallback],
                        verbose=1,
                        validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=1)

    model.save('trained_model')
    model.save_weights('trained_model_weights')
    with open('trained_model_json', 'w') as jsonfile:
        jsonfile.write(model.to_json())

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    fig = pyplot.figure()
    val_acc = fig.add_subplot(121)
    val_loss = fig.add_subplot(122)
    val_acc.plot(history.history['val_acc'], label='Validation Accuracy')
    val_loss.plot(history.history['val_loss'], label='Validation Loss')
    val_acc.set_xlabel("Epoch")
    val_acc.set_ylabel("Validation Accuracy")
    val_loss.set_ylabel("Validation Accuracy")
    pyplot.legend()
    pyplot.show(fig)


def EvaluateOneFile(wavFileName='resources/f2cnn/TRAIN/DR3.FCKE0.SI1111.WAV'):
    from keras.models import  load_model

    print("File:\t\t{}".format(wavFileName))
    # #### READING CONFIG FILE
    config = ConfigParser()
    config.read('F2CNN.conf')
    framerate = config.getint('FILTERBANK', 'FRAMERATE')
    nchannels = config.getint('FILTERBANK', 'NCHANNELS')
    lowcutoff = config.getint('FILTERBANK', 'LOW')

    ustos=1/1000000.
    # ##### PREPARATION OF FILTERBANK
    # CENTER FREQUENCIES ON ERB SCALE
    CENTER_FREQUENCIES = filters.centre_freqs(framerate, nchannels, lowcutoff)
    # Filter coefficient for a Gammatone filterbank
    FILTERBANK_COEFFICIENTS = filters.make_erb_filters(framerate, CENTER_FREQUENCIES)

    filtered, framerate = GetFilteredOutputFromFile(wavFileName, FILTERBANK_COEFFICIENTS)
    fbPath = os.path.splitext(wavFileName)[0] + '.FB'
    phnPath = os.path.splitext(wavFileName)[0] + '.PHN'
    formants, sampPeriod = ExtractFBFile(fbPath)
    envelope = ExtractEnvelopeFromMatrix(filtered)
    phonemes = ExtractPhonemes(phnPath)
    print(phonemes)
    nb = int(len(envelope[0]) - 0.11*framerate)
    input_data = numpy.zeros([nb, 11, 128])
    print(input_data.shape)
    START = int(0.055 * framerate)
    STEP = int(framerate*sampPeriod*ustos)
    for i in range(0, nb):
        input_data[i] = [[channel[START + i + (k - 5) * STEP] for channel in envelope] for k in range(11)]
    for i, matrix in enumerate(input_data):
        input_data[i] = normalizeInput(matrix)
    input_data.astype('float32')
    model = load_model('trained_model')
    # scores = model.evaluate(input_data, )
    scores = model.predict(input_data.reshape(nb, 11, 128, 1))
    rising = [-100 if neg > pos else 200 for neg, pos in scores]
    falling = [200 if neg > pos else -100 for neg, pos in scores]
    PlotEnvelopesAndCNNResultsWithPhonemes(envelope, rising, falling, CENTER_FREQUENCIES, phonemes, formants, sampPeriod, wavFileName)
    print("\t\t{}\tdone !".format(wavFileName))


def EvaluateRandom(testMode=False):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence tensorflow logs

    TotalTime = time.time()

    if not os.path.isdir("graphs"):
        os.mkdir(os.path.join('graphs','FallingOrRising'))
    if testMode:
        # # Test Files
        wavFiles = glob.glob(os.path.join('testFiles', '*.WAV'))
    else:
        # Get all the WAV files under resources/fcnn
        wavFiles = glob.glob(os.path.join('resources', 'f2cnn', '*', '*.WAV'))
    print("\n###############################\nEvaluating network on WAV files in '{}'.".format(os.path.split(wavFiles[0])[0]))

    if not wavFiles:
        print("NO WAV FILES FOUND")
        exit(-1)
    numpy.random.shuffle(wavFiles)
    for file in wavFiles:
        EvaluateOneFile(file)

    print("Evaluating network on all files.")
    print('              Total time:', time.time() - TotalTime)
    print('')
