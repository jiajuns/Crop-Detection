import h5py
from keras.models import Model
from keras.layers import Input, Activation, Concatenate
from keras.layers import Flatten, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import AveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers

def SqueezeNet(nb_classes, inputs=(3, 224, 224), learning_rate=0.00001, is_testing=False):
    """ Keras Implementation of SqueezeNet(arXiv 1602.07360)
    @param nb_classes: total number of final categories
    Arguments:
    inputs -- shape of the input images (channel, cols, rows)
    """

    input_img = Input(shape=inputs)
    conv1 = Convolution2D(
        96, (7, 7), activation='relu', kernel_initializer='glorot_uniform',
        strides=(2, 2), padding='same', name='conv1',
        data_format="channels_first")(input_img)
    maxpool1 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool1',
        data_format="channels_first")(conv1)
    fire2_squeeze = Convolution2D(
        16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_squeeze',
        data_format="channels_first")(maxpool1)
    fire2_expand1 = Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_expand1',
        data_format="channels_first")(fire2_squeeze)
    fire2_expand2 = Convolution2D(
        64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_expand2',
        data_format="channels_first")(fire2_squeeze)
    merge2 = Concatenate(axis=1)([fire2_expand1, fire2_expand2])

    fire3_squeeze = Convolution2D(
        16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_squeeze',
        data_format="channels_first")(merge2)
    fire3_expand1 = Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_expand1',
        data_format="channels_first")(fire3_squeeze)
    fire3_expand2 = Convolution2D(
        64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_expand2',
        data_format="channels_first")(fire3_squeeze)
    merge3 = Concatenate(axis=1)([fire3_expand1, fire3_expand2])

    fire4_squeeze = Convolution2D(
        32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_squeeze',
        data_format="channels_first")(merge3)
    fire4_expand1 = Convolution2D(
        128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_expand1',
        data_format="channels_first")(fire4_squeeze)
    fire4_expand2 = Convolution2D(
        128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_expand2',
        data_format="channels_first")(fire4_squeeze)
    merge4 = Concatenate(axis=1)([fire4_expand1, fire4_expand2])
    maxpool4 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool4',
        data_format="channels_first")(merge4)

    fire5_squeeze = Convolution2D(
        32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_squeeze',
        data_format="channels_first")(maxpool4)
    fire5_expand1 = Convolution2D(
        128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_expand1',
        data_format="channels_first")(fire5_squeeze)
    fire5_expand2 = Convolution2D(
        128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_expand2',
        data_format="channels_first")(fire5_squeeze)
    merge5 = Concatenate(axis=1)([fire5_expand1, fire5_expand2])

    fire6_squeeze = Convolution2D(
        48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_squeeze',
        data_format="channels_first")(merge5)
    fire6_expand1 = Convolution2D(
        192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_expand1',
        data_format="channels_first")(fire6_squeeze)
    fire6_expand2 = Convolution2D(
        192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_expand2',
        data_format="channels_first")(fire6_squeeze)
    merge6 = Concatenate(axis=1)([fire6_expand1, fire6_expand2])

    fire7_squeeze = Convolution2D(
        48, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire7_squeeze',
        data_format="channels_first")(merge6)
    fire7_expand1 = Convolution2D(
        192, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire7_expand1',
        data_format="channels_first")(fire7_squeeze)
    fire7_expand2 = Convolution2D(
        192, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire7_expand2',
        data_format="channels_first")(fire7_squeeze)
    merge7 = Concatenate(axis=1)([fire7_expand1, fire7_expand2])

    fire8_squeeze = Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire8_squeeze',
        data_format="channels_first")(merge7)
    fire8_expand1 = Convolution2D(
        256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire8_expand1',
        data_format="channels_first")(fire8_squeeze)
    fire8_expand2 = Convolution2D(
        256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire8_expand2',
        data_format="channels_first")(fire8_squeeze)
    merge8 = Concatenate(axis=1)([fire8_expand1, fire8_expand2])

    maxpool8 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool8',
        data_format="channels_first")(merge8)
    fire9_squeeze = Convolution2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire9_squeeze',
        data_format="channels_first")(maxpool8)
    fire9_expand1 = Convolution2D(
        256, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire9_expand1',
        data_format="channels_first")(fire9_squeeze)
    fire9_expand2 = Convolution2D(
        256, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire9_expand2',
        data_format="channels_first")(fire9_squeeze)
    merge9 = Concatenate(axis=1)([fire9_expand1, fire9_expand2])

    fire9_dropout = Dropout(0.5, name='fire9_dropout')(merge9)
    conv10 = Convolution2D(
        nb_classes, (1, 1), kernel_initializer='glorot_uniform',
        padding='valid', name='conv10',
        data_format="channels_first")(fire9_dropout)
    
    # The size should match the output of conv10
    avgpool10 = AveragePooling2D(
        (13, 13), name='avgpool10',
        data_format="channels_first")(conv10)

    flatten = Flatten(name='flatten')(avgpool10)
    softmax = Activation("softmax", name='softmax')(flatten)

    model = Model(inputs=input_img, outputs=softmax)

    filepath="weights.best.hdf5"

    if is_testing:
        model.load_weights("weights.best.hdf5")
        callbacks_list = None
    else:
        adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        print('squeezeNet model summary: ')
        print(model.summary())

        # checkpoint
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

    return model, callbacks_list



def ConvNet(nb_classes, inputs=(3, 224, 224), learning_rate=0.00001, reg=0.001, is_testing=False):
    """ Keras Implementation of SqueezeNet(arXiv 1602.07360)
    @param nb_classes: total number of final categories
    Arguments:
    inputs -- shape of the input images (channel, cols, rows)
    """
    model = Sequential()
    model.add(Convolution2D(
        96, (7, 7), activation='relu', kernel_initializer='glorot_uniform',
        strides=(2, 2), padding='same', name='conv1', input_shape=inputs,
        data_format="channels_first"))
    model.add(Convolution2D(
        96, (5, 5), activation='relu', kernel_initializer='glorot_uniform',
        strides=(2, 2), padding='same', name='conv2',
        data_format="channels_first"))
    model.add(Convolution2D(
        128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        strides=(2, 2), padding='same', name='conv3',
        data_format="channels_first"))
    model.add(MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool1',
        data_format="channels_first"))
    model.add(Dropout(0.5, name='dropout'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(nb_classes, kernel_regularizer=regularizers.l2(reg), activation='softmax'))
    
    filepath="simple_weights.best.hdf5"

    if is_testing:
        adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam,
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        model.load_weights("simple_weights.best.hdf5")
        callbacks_list = None
    else:
        adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        print('conv model summary: ')
        print(model.summary())

        # checkpoint
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

    return model, callbacks_list


def ConvNet_gray(nb_classes, inputs=(1, 224, 224), learning_rate=0.00001, reg=0.001, is_testing=False):
    """ Keras Implementation of SqueezeNet(arXiv 1602.07360)
    @param nb_classes: total number of final categories
    Arguments:
    inputs -- shape of the input images (channel, cols, rows)
    """
    model = Sequential()
    model.add(Convolution2D(
        96, (7, 7), activation='relu', kernel_initializer='glorot_uniform',
        strides=(2, 2), padding='same', name='conv1', input_shape=inputs,
        data_format="channels_first"))
    model.add(Convolution2D(
        96, (5, 5), activation='relu', kernel_initializer='glorot_uniform',
        strides=(2, 2), padding='same', name='conv2',
        data_format="channels_first"))
    model.add(Convolution2D(
        128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        strides=(2, 2), padding='same', name='conv3',
        data_format="channels_first"))
    model.add(MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool1',
        data_format="channels_first"))
    model.add(Dropout(0.5, name='dropout'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(nb_classes, kernel_regularizer=regularizers.l2(reg), activation='softmax'))
    
    filepath="simple_gray_weights.best.hdf5"

    if is_testing:
        adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam,
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        model.load_weights("simple_gray_weights.best.hdf5")
        callbacks_list = None
    else:
        adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        print('conv model summary: ')
        print(model.summary())

        # checkpoint
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

    return model, callbacks_list