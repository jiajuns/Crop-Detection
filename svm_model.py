from keras.models import Sequential
from keras.layers.core import Dense
from keras import regularizers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

def svm(input_size, reg=0.01, learning_rate=0.00001):
    model = Sequential()
    model.add(Dense(2, input_shape=(input_size,),
                    kernel_regularizer=regularizers.l2(reg)))

    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer='rmsprop',
          loss='categorical_crossentropy',
          metrics=['accuracy'])
    
    filepath="SVM_weights.best.hdf5"
    # checkpoint
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    print('SVM model summary: ')
    print(model.summary())
    return model, callbacks_list

