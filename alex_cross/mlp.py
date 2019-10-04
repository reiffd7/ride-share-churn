import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
import pandas as pd

import os
os.environ['TF_KERAS']='1'

from keras_radam import RAdam

os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # disable Tensorflow warnings

def def_mlp_model(X, nn_hl, activ='sigmoid', opt = 'adam'):
    """ Defines hidden layer mlp model
        X is the training array
        nn_hl is the desired number of neurons in the hidden layer
        activ is the activation function (could be 'tanh', 'relu', etc.)
    """
    num_coef = X.shape[1]
    model = Sequential() # sequential model is a linear stack of layers
    model.add(Dense(units=nn_hl,
                    input_shape=(num_coef,),
                    activation=activ,
                    use_bias=True,
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros',
                    kernel_regularizer=None,
                    bias_regularizer=None,
                    activity_regularizer=None,
                    kernel_constraint=None,
                    bias_constraint=None))
    model.add(Dense(units=1,
                    activation=activ,
                    use_bias=True,
                    kernel_initializer='glorot_uniform'))
    sgd = SGD(lr=1.0, decay=1e-7, momentum=.9) # using stochastic gradient descent
    model.compile(loss="mean_squared_error", optimizer=opt, metrics=["mse"] )
    return model
if __name__ == "__main__":
    df = pd.read_csv('cleaned_data.csv', index_col=0)
    
    # drop outliers
    df = df[df['avg_dist'] < 50]
    df = df[df['trips_in_first_30_days'] < 30]
    
    y = df.pop('churn')
    X = df
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)
    
    # Create a Sequential model
    model = Sequential()

    # Add an input layer and a hidden layer with 10 neurons
    model.add(Dense(104, input_shape=(13,), activation="tanh"))
    model.add(Dropout(.15))

    model.add(Dense(52, input_shape=(13,), activation="tanh"))
    model.add(Dropout(.15))

    # model.add(Dense(26, activation="relu"))
    # model.add(Dropout(.15))

    opt_R = RAdam(lr = 0.01, beta_1=0.8, total_steps=10000, warmup_proportion=0.1, min_lr=1e-4)
     # Add a 1-neuron output layer
    model.add(Dense(1, activation='sigmoid'))
    opt = Adam(learning_rate=0.005)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy', 'binary_accuracy'])
    model.fit(X_train, y_train, batch_size=50, epochs=10,
        verbose=1, validation_data=(X_test, y_test))
    model.evaluate(X_test, y_test, batch_size=50)
    # Summarise your model
    model.summary()
    predictions = model.predict(X_test, batch_size=50)
    
    
    print(f'Prediction: ')
    
    # score = model.evaluate(X_test, y_test, verbose=0)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1]) # this is the one we care about