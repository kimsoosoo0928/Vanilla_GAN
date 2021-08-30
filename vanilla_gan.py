from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 127.5 - 1
x_test = x_test / 127.5 - 1

print(x_train.min(), x_train.max())
# -1.0 1.0 

x_train = x_train.reshape(-1, 28 * 28)
print(x_train.shape)
# (60000, 784) 

# hyperparameters

NOISE_DIM = 10
adam = Adam(lr=0.0002, beta_1=0.5)

# generator

generator = Sequential([
    Dense(256, input_dim=NOISE_DIM),
    LeakyReLU(0.2),
    Dense(512),
    LeakyReLU(0.2),
    Dense(1024),
    LeakyReLU(0.2),
    Dense(28*28, activation='tanh')
])

generator.summary()

discriminator = Sequential([
    Dense(1024, input_shape=(28*28,), kernel_initializer=RandomNormal(stddev=0.02)),
    LeakyReLU(0.2),
    Dropout(0.3),
    Dense(512),
    LeakyReLU(0.2),
    Dropout(0.3),
    Dense(256),
    LeakyReLU(0.2),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

discriminator.summary()

discriminator.compile(loss='binary_crossentropy', optimizer=adam)

