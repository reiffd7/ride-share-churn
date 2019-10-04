import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import pandas as pd

import os
os.environ['TF_KERAS']='1'

from keras_radam import RAdam

# Create function returning a compiled network
def create_mlp():
    
    # Start neural network
    model = Sequential()
    
    # Add initial layer
    model.add(Dense(52, activation='relu', input_shape=(13,)))
    model.add(Dropout(0.1))
    
    # Add hidden layers
    model.add(Dense(13, activation='relu'))
    model.add(Dropout(0.1))
    # Add final layer with a sigmoid activation function
    model.add(Dense(1, activation='sigmoid'))

    # Compile neural network
    model.compile(loss='binary_crossentropy', # Cross-entropy
                    optimizer='adam', # Root Mean Square Propagation
                    metrics=['accuracy']) # Accuracy performance metric
    
    # Return compiled network
    return model

if __name__ == "__main__":
    df = pd.read_csv('cleaned_data.csv', index_col=0)
    
    # drop outliers
    df = df[df['avg_dist'] < 50]
    df = df[df['trips_in_first_30_days'] < 30]
    
    y = df.pop('churn')
    X = df
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)
    
    # # Create a Sequential model
    # model = Sequential()

    # # Add an input layer and a hidden layer with 10 neurons
    # model.add(Dense(104, input_shape=(13,), activation="tanh"))
    # model.add(Dropout(.15))

    # model.add(Dense(52, input_shape=(13,), activation="tanh"))
    # model.add(Dropout(.15))

    # # model.add(Dense(26, activation="relu"))
    # # model.add(Dropout(.15))

    # opt_R = RAdam(lr = 0.01, beta_1=0.8, total_steps=10000, warmup_proportion=0.1, min_lr=1e-4)
    #  # Add a 1-neuron output layer
    # model.add(Dense(1, activation='sigmoid'))
    # opt = Adam(learning_rate=0.005)
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy', 'binary_accuracy'])
    # model.fit(X_train, y_train, batch_size=50, epochs=10,
    #     verbose=1, validation_data=(X_test, y_test))
    # model.evaluate(X_test, y_test, batch_size=50)
    # # Summarise your model
    # model.summary()
    # predictions = model.predict(X_test, batch_size=50)
    # print(f'Prediction: ')



    # Wrap Keras model so it can be used by scikit-learn
    neural_network = KerasClassifier(build_fn=create_mlp, 
                                    epochs=10, 
                                    batch_size=50, 
                                    verbose=0)

    # Evaluate neural network using three-fold cross-validation
    print(cross_val_score(neural_network, X_train, y_train,scoring='accuracy', cv=3))
    
    
    
