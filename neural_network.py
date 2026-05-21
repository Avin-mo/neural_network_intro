import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import random

# --------------------------------------------------
# POLYNOMIAL NEURAL NETWORK
# Architecture D = (4, 2, 1)
# Activation function: sigma(x) = x^3
# Learn:
# f(x,y,z,w) = x^3 + y^3 + z^3
# INSPO CODE FROM: https://www.geeksforgeeks.org/deep-learning/neural-networks-a-beginners-guide/
# --------------------------------------------------

# ACTIVATION FUNCTION
def cubic_activation(x):
    return tf.math.pow(x, 3)

    
# --------------------------------------------------
# GENERATE TRAINING DATA
# --------------------------------------------------
# set seeds for consistency
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# 1000 random samples
X = np.random.uniform(-1, 1, size=(1000, 4))

# target values:
# f(x,y,z,w)=x^3+y^3+z^3
y = (
    X[:, 0]**3 +
    X[:, 1]**3 +
    X[:, 2]**3
)

# --------------------------------------------------
# NEURAL NETWORK
# D = (4,2,1)
# --------------------------------------------------

model = Sequential()

# hidden layer: 4 -> 2
model.add(Dense(2, input_dim=4, activation=cubic_activation))

# output layer: 2 -> 1
model.add(Dense(1))


# --------------------------------------------------
# COMPILING
# --------------------------------------------------

model.compile(loss='mse', optimizer='adam')


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

model.fit(
    X,
    y,
    epochs=200,
    batch_size=25,
    verbose=1
)


# --------------------------------------------------
# TESTING
# --------------------------------------------------

test_data = np.array([[1, 2, 3, 5]])

prediction = model.predict(test_data)

print("\nPrediction:", prediction)

# actual value:
# 1^3 + 2^3 + 3^3 = 36

print("Actual:", 1**3 + 2**3 + 3**3)