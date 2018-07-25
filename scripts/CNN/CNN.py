from __future__ import print_function

import csv
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy
from matplotlib import pyplot

from scripts.LabelDataGenerator import GetLabels


def SeparateTestTrain(pathToInput, pathToLabel):
    x = [[], []]
    y = [[], []]
    CASE = ['TEST', 'TRAIN']
    currentCase = 0
    lastRegion = 'DR1'
    input_data = numpy.load(pathToInput)
    label_data = GetLabels(True)
    running = True
    with open(pathToLabel, 'r') as labels:
        reader = csv.reader(labels)
        for i, (region, speaker, sentence, phoneme, timepoint, slope, pvalue, sign) in enumerate(reader):
            if region != lastRegion:  # If there is a change in region
                if region == 'DR1':  # And the change is to a DR1
                    print('AH')
                    currentCase += 1  # It means we transition form TEST to TRAIN
                lastRegion = region
            x[currentCase].append(input_data[i])
            y[currentCase].append(int(sign))
    return numpy.array(x[0]), numpy.array(y[0]), numpy.array(x[1]), numpy.array(y[1])


batch_size = 128
num_classes = 2
epochs = 20

# input image dimensions
img_rows, img_cols = 11, 128

x_test, y_test, x_train, y_train = SeparateTestTrain('../../trainingData/input_data.npy',
                                                     '../../trainingData/label_data.csv')

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# TODO: normalize
# x_train /= 255
# x_test /= 255

print(x_test[0])
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

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
stopCallback = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.001, patience=1, verbose=1, mode='min',
                                             baseline=None)
batch_loss = []


class BatchPlotter(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.loss = []

    def on_batch_end(self, batch, logs=None):
        self.loss.append(logs['loss'])


batchPlotCallback = BatchPlotter()

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=[stopCallback, batchPlotCallback],
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
loss = history.history['loss']
line, =pyplot.plot(batchPlotCallback.loss, label='Loss function')
pyplot.xlabel("batch number")
pyplot.ylabel("Loss")
pyplot.legend()
pyplot.show()
