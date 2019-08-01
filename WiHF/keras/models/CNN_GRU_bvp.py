# -*- coding: utf-8 -*-
"""Code to initialize the architecture of a Domain-Adaptive Recurrent Neural Network (DA-RNN) for driving manoeuver
anticipation. 

From the paper "Robust and Subject-Independent Driving Manoeuvre Anticipation through Domain-Adversarial 
Recurrent Neural Networks", by Tonutti M, Ruffaldi E, et al.

Author: Michele Tonutti
Date: 2019-02-09
"""
import keras
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.callbacks import EarlyStopping
from keras.models import Model, Sequential
from keras.layers import Input, Dense, TimeDistributed, Flatten, Reshape, Permute, Dropout
from keras.layers import LSTM, GRU
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import confusion_matrix

import numpy as np

BIAS_INITIALIZER = 'ones'
KERNEL_INITIALIZER = 'VarianceScaling'
# OPTIMIZER = 'adam'
OPTIMIZER = 'rmsprop'
UNITS_GRU = 128


class CNN_GRU_bvp:
    """Class to create the architecture of a Domain Adaptive RNN for maneuver anticipation.
    
    Parameters
    ----------
    n_timesteps : int
        Number of timesteps in each sample.
    n_feats_head : int
        Number of features related to head movements.
    n_feats_outside : int
        Number of features from the camera outside the car (street).
    n_feats_gaze : int
        Number of features related to gaze movements.
    dense_units : int, optional
        Number of units in the dense layers.
    gru_units : int, optional
        Number of units in the GRU layers.
    lstm_units : int, optional
        Number of units in the LSTM layers.
    lambda_reversal_strength : float, optional
        Constant controlling the ratio of the domain classifier loss to action classifier loss 
        (lambda = L_class / L_domain)
        A higher lambda will increase the influence of the domain classifier, rewarding domain-invariant features. 
        A lower lambda will increase the influence of the manoeuver anticipation, rewarding correct classification.
    kernel_initializer : str, optional
        Initializer for the kernel of recurrent layers.
    bias_initializer: str, optional
        Initializer for the bias of recurrent layers.
    dropout : float, optional
        Strength of regular dropout in Dense layers.
    rec_dropout: float, optional
        Strength of recurrent dropout in recurrent layers.
    """

    def __init__(self,
                 n_timesteps,
                 n_features_row,
                 n_features_column,
                 n_batch_size,
                 n_features_channel,
                 n_epochs,
                 n_patience,
                 n_classes,
                 save_model_path,
                 n_gru_units=UNITS_GRU):
        self.n_timesteps = n_timesteps
        self.n_classes = n_classes
        self.n_batch_size = n_batch_size
        self.n_epochs = n_epochs
        self.n_patience = n_patience
        self.input_shape_cnn = (n_features_row, n_features_column,
                                n_features_channel)
        self.input_shape_rnn = (n_timesteps, n_features_row, n_features_column,
                                n_features_channel)

        self.gru_units = n_gru_units
        self.save_model_path = save_model_path
        self.model = None

    def create_architecture(self):
        """Create the model architecture.
        
        Returns
        -------
        type : keras.Model
            The initialized model object.
        """
        cnn = Sequential()
        cnn.add(
            Conv2D(16,
                   kernel_size=(5, 5),
                   activation='relu',
                   input_shape=self.input_shape_cnn))
        cnn.add(MaxPooling2D(pool_size=(2, 2)))  #16
        cnn.add(Flatten())
        cnn.add(Dense(64, activation='relu'))  #64
        cnn.add(Dropout(0.5))
        cnn.add(Dense(64, activation='relu'))  #64

        # Model Definition
        model = Sequential()
        model.add(TimeDistributed(cnn, input_shape=self.input_shape_rnn))
        model.add(GRU(self.gru_units))
        model.add(Dropout(0.2))
        model.add(Dense(self.n_classes, activation='softmax'))

        self.model = model

        return self.model

    def compile_model(self,
                      loss='categorical_crossentropy',
                      optimizer=OPTIMIZER,
                      metrics=None):
        """Compile the model.
        
        Parameters
        ----------
        loss : str or custom loss function, optional
            Loss function to use for the training. Categorical crossentropy by default.
        optimizer : str or custom optimizer object, optional
            Optimizer to use for the training. Adam by default.
        metrics : list
            Metric to use for the training. Can be a custom metric function.
        loss_weights: dict
            Dictionary of loss weights. The items of the dictionary can be lists, with one weight per timestep.

        Returns
        -------
        type : keras.Model
            The compiled model.
        """
        if metrics is None:
            metrics = ['accuracy']

        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        return self.model

    def fit_model(self, train_data, train_label):
        """Fit the model on a training set.
        
        The training sets must be divided in batches that contain both the source and target domain, 
        in order to perform adaptation.
                    
        Parameters
        ----------
        X_train_head : np.ndarray
            Head features. Shape = (n_samples, n_timestamps, n_features)
        X_train_outside : np.ndarray
            Context features. Shape = (n_samples, n_timestamps, n_features)
        X_train_gaze : np.ndarray
            Gaze features. Shape = (n_samples, n_timestamps, n_features)
        y_train : np.ndarray
            Action labels, encoded as integers.
        y_train_domain : np.ndarray
            Binary domain labels, encoded as integers.
        batch_size : int, optional
            Size of the batches for training. Default = 128.
        epochs : int, optional
            Number of epochs to run the training. Default = 1000.
        patience: int, optional
            Number of epochs to wait without an improvement in the validation loss for early stopping. Default = 30.
            
        Returns
        -------
        type : keras.History
            The history of the trained model as a keras.History object.
        """

        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=self.n_patience)

        model_history = self.model.fit(train_data,
                                       train_label,
                                       batch_size=self.n_batch_size,
                                       epochs=self.n_epochs,
                                       verbose=2,
                                       validation_split=0.1,
                                       shuffle=True,
                                       callbacks=[early_stopping])

        return model_history

    def predict_proba(self, test_data, test_label):
        """Predict probabilities on a test set.
        
        The test set can come from either the source or target domain.
        
        Parameters
        ----------
        X_test_head : np.ndarray
            Head features. Shape = (n_samples, n_timestamps, n_features)
        X_test_outside : np.ndarray
            Context features. Shape = (n_samples, n_timestamps, n_features)
        X_test_gaze : np.ndarray
            Gaze features. Shape = (n_samples, n_timestamps, n_features)

        Returns
        -------
        type : np.ndarray
            Predicted probabilities.
        """
        scores = self.model.evaluate(test_data, test_label, verbose=0)
        test_label_predict = self.model.predict(test_data)
        test_label_predict = np.argmax(test_label_predict, axis=-1) + 1
        test_label = np.argmax(test_label, axis=-1) + 1
        cm = confusion_matrix(test_label, test_label_predict)
        print(cm)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(cm)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

        # Save model to file
        print("Saving model to disk \n")
        self.model.save(self.save_model_path)

        return [cm, scores]
