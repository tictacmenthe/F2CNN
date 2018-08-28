"""
This file includes code allowing the training of the neural networks,
and every helper function needed to pack/unpack input and label data.
"""
import csv
from configparser import ConfigParser
from os.path import join

import numpy
from matplotlib import pyplot


def normalizeInput(matrix: numpy.ndarray):
    minvalue, maxvalue = matrix.min(), matrix.max()
    if minvalue > maxvalue:
        raise ValueError("minvalue must be less than or equal to maxvalue")
    elif minvalue <= 0:
        print(matrix.shape)
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
    print(input_data.shape)

    with open(pathToLabel, 'r') as labels:
        reader = csv.reader(labels)
        for i, (test, region, speaker, sentence, phoneme, timepoint, slope, pvalue, sign) in enumerate(reader):
            if test == 'TEST':
                x[0].append(input_data[i])
                y[0].append(int(sign))
            else:
                x[1].append(input_data[i])
                y[1].append(int(sign))
    return numpy.array(x[0]), numpy.array(y[0]), numpy.array(x[1]), numpy.array(y[1])


def TrainAndPlotLoss(inputFile=None):
    """
    Trains the CNN suing the given input FIle
    :param inputFile: path to a .npy file tensor of Nx11x128 values
    """
    import keras

    # ### CONFIGURATION
    config = ConfigParser()
    config.read('F2CNN.conf')
    batch_size = config.getint('CNN', 'BATCHSIZE')
    num_classes = config.getint('CNN', 'CLASSES')
    epochs = config.getint('CNN', 'EPOCHS')
    # input image dimensions
    img_rows, img_cols = config.getint('CNN', 'RADIUS')*2+1, config.getint('FILTERBANK', 'NCHANNELS')  # 11x128 default

    inputPath = inputFile or join('trainingData', 'last_input_data.npy')  # default file if none provided
    labelPath = join('trainingData', 'label_data.csv')

    x_test, y_test, x_train, y_train = SeparateTestTrain(inputPath, labelPath)

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
    print('None test:', len([sign for sign in y_test if sign == 2]))
    print('Rising train:', len([sign for sign in y_train if sign == 1]))
    print('Falling train:', len([sign for sign in y_train if sign == 0]))
    print('None train:', len([sign for sign in y_train if sign == 2]))

    print(x_train.shape, 'train samples')
    print(x_test.shape, 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print("Categories: [falling, rising]")

    # #### KERAS MODEL BUILDING
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same',
                                  input_shape=x_train.shape[1:]))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(32, (3, 3)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(64, (3, 3), padding='same'))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Conv2D(64, (3, 3)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(516))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes))
    model.add(keras.layers.Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])

    # STOP callback, used to stop training before the maximum number of epochs,
    # if the network stops getting better for the value 'monitor',
    # with less than 'min_delta' variation over 'patience' epochs
    stopCallback = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.01, patience=5, verbose=1, mode='auto',
                                                 baseline=None)

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[stopCallback],
                        verbose=1,
                        validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=1)

    print("Model saved as a keras file 'last_trained_model'.")
    model.save('last_trained_model')

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # Plotting of the training results, validation accuracy and validation loss accross epochs
    fig = pyplot.figure(figsize=(32,16))
    val_acc = fig.add_subplot(121)
    val_loss = fig.add_subplot(122)
    val_acc.plot(history.history['val_acc'], label='Validation Accuracy')
    val_loss.plot(history.history['val_loss'], label='Validation Loss')
    val_acc.set_xlabel("Epoch")
    val_acc.set_ylabel("Validation Accuracy")
    val_loss.set_ylabel("Validation Loss")
    val_acc.legend()
    val_loss.legend()
    pyplot.show(fig)
    pyplot.savefig('last_trained_model_results.png')
