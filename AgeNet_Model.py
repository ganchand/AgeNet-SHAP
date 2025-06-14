import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta
from tensorflow.keras.regularizers import l2

def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    model.add(Dense(256, activation='relu',
                    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                    kernel_regularizer=tf.keras.regularizers.l2(0.02)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(128, activation='relu',
                    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                    kernel_regularizer=tf.keras.regularizers.l2(0.02)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(64, activation='relu',
                    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                    kernel_regularizer=tf.keras.regularizers.l2(0.02)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(32, activation='relu',
                    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                    kernel_regularizer=tf.keras.regularizers.l2(0.02)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(1, activation='linear',
                    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)))
    
    model.summary()
    return model
