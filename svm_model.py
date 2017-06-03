from keras.models import Sequential
from keras.layers.core import Dense
from keras import regularizers

def svm(input_size, reg=0.001):
    model = Sequential()
    model.add(Dense(2, input_shape=(input_size,),
                    kernel_regularizer=regularizers.l2(reg)))
    model.compile(optimizer='rmsprop',
          loss='hinge',
          metrics=['accuracy'])

    print('SVM model summary: ')
    print(model.summary())
    return model

