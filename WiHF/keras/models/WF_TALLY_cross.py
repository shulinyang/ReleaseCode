from __future__ import print_function
import keras
from keras import backend as K
K.set_image_dim_ordering('tf')
import numpy as np
from math import exp
import os
import time
import keras.layers as kl
from keras.models import Model, Sequential, load_model
from keras.layers import (Input, LSTM, Dense, concatenate, Dropout, GRU,
                          Masking)
from keras.callbacks import EarlyStopping

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from models.Gradient_Reverse_Layer import GradientReversal
from keras_attention_block import *

BIAS_INITIALIZER = 'ones'
KERNEL_INITIALIZER = 'VarianceScaling'
OPTIMIZER = 'rmsprop'

UNITS_DENSE = 128
UNITS_GRU = 128
UNITS_LSTM = 64


class WF_TALLY_cross(object):
    def __init__(self,
                 n_timesteps,
                 n_features_row,
                 n_features_column,
                 n_features_channel,
                 save_model_path,
                 gradient_reversal_rate=0.0,
                 n_classes_main=9,
                 n_classes_aux=9,
                 n_features=32,
                 n_batch_size=32,
                 n_epochs=1000,
                 n_patience=30,
                 n_gru_units=UNITS_GRU,
                 kernel_initializer=KERNEL_INITIALIZER,
                 grl='auto'):
        ## Set Defualts
        self.n_timesteps = n_timesteps
        self.n_features_row = n_features_row
        self.n_features_column = n_features_column
        self.n_features_channel = n_features_channel
        self.n_classes_main = n_classes_main
        self.n_classes_aux = n_classes_aux
        self.n_features = n_features
        self.n_batch_size = n_batch_size
        self.n_epochs = n_epochs
        self.n_patience = n_patience
        self.gradient_reversal_rate = gradient_reversal_rate

        self.gru_units = n_gru_units
        self.save_model_path = save_model_path
        self.kernel_initializer = kernel_initializer

        self.domain_invariant_features = None
        self.input_shape = (n_features_row, n_features_column,
                            n_features_channel)
        self.input_shape_rnn = (n_timesteps, n_features_row, n_features_column,
                                n_features_channel)
        self.grl = grl
        # Set reversal gradient value.
        if grl is 'auto':
            self.grl_rate = 1.0
        else:
            self.grl_rate = grl

        # Build the model
        self.model = self._build()

    def feature_extractor_aux(self, inp, name):
        ''' 
		This function defines the structure of the feature extractor part.
		'''
        cnn = Sequential()
        cnn.add(
            kl.Conv2D(16,
                      kernel_size=(3, 3),
                      activation='relu',
                      input_shape=self.input_shape))
        cnn.add(kl.Flatten())
        cnn.add(kl.Dense(64, activation='relu'))  #64
        cnn.add(kl.Dropout(0.5))
        cnn.add(kl.Dense(64, activation='relu'))  #64

        out = kl.TimeDistributed(cnn, name=name)(inp)
        out = kl.GRU(self.gru_units)(out)
        feature_output = kl.Dropout(0.2)(out)
        self.domain_invariant_features = feature_output
        return feature_output

    def feature_extractor_main(self, inp, name):
        ''' 
		This function defines the structure of the feature extractor part.
		'''
        cnn = Sequential()
        cnn.add(
            kl.Conv2D(16,
                      kernel_size=(3, 3),
                      activation='relu',
                      input_shape=self.input_shape))
        cnn.add(kl.Flatten())
        cnn.add(kl.Dense(64, activation='relu'))  #64
        cnn.add(kl.Dropout(0.5))
        cnn.add(kl.Dense(self.n_features, activation='relu'))  #64

        out = kl.TimeDistributed(cnn, name=name)(inp)
        # out = SelfAttention1DLayer(similarity="linear", dropout_rate=0.2)(out)
        out = kl.GRU(self.gru_units)(out)
        feature_output = kl.Dropout(0.2)(out)
        self.domain_invariant_features = feature_output
        return feature_output

    def classifier_aux(self, inp, output_size, name):
        ''' 
		This function defines the structure of the classifier part.
		'''
        # out = kl.Dropout(0.2)(inp)
        classifier_output = kl.Dense(output_size,
                                     activation="softmax",
                                     name=name)(inp)
        return classifier_output

    def classifier_main(self, inp_feature_main, inp_feature_aux, output_size,
                        name):
        ''' 
		This function defines the structure of the classifier_main part.
		'''
        # resize_classifier = kl.Lambda(
        #     K.tile,
        #     arguments={'n': (1, self.gru_units // self.n_classes_gesture)
        #                })(inp_classifier)
        # out = keras.layers.concatenate(
        #     [inp_feature_gesture, resize_classifier], axis=-1)
        flip_layer = GradientReversal(self.gradient_reversal_rate)
        inp_feature_aux = flip_layer(inp_feature_aux)
        out = keras.layers.concatenate([inp_feature_main, inp_feature_aux],
                                       axis=-1)
        out = kl.Dense(128, activation="relu")(out)
        out = kl.Dropout(0.5)(out)
        classifier_main_output = kl.Dense(output_size,
                                          activation="softmax",
                                          name=name)(out)
        return classifier_main_output

    def _build(self):
        '''
		This function builds the network based on the Feature Extractor, Classifier_aux and Classifier_main parts.
		'''
        inp_1 = Input(shape=self.input_shape_rnn, name="main_input_1")
        inp_2 = Input(shape=self.input_shape_rnn, name="main_input_2")
        feature_output_aux = self.feature_extractor_aux(
            inp_1, 'feature_output_aux')
        feature_output_main = self.feature_extractor_main(
            inp_2, 'feature_output_main')

        # classifier_output_aux = self.classifier_aux(feature_output_aux,
        #                                             self.n_classes_aux,
        #                                             'classifier_output_aux')
        classifier_output_aux = self.classifier_main(feature_output_aux,
                                                     feature_output_main,
                                                     self.n_classes_aux,
                                                     'classifier_output_aux')
        classifier_output_main = self.classifier_main(
            feature_output_main, feature_output_aux, self.n_classes_main,
            'classifier_output_main')

        model = keras.models.Model(inputs=[inp_1, inp_2],
                                   outputs=[
                                       classifier_output_aux,
                                       classifier_output_main,
                                   ])
        return model

    def retrieve_model(self):
        print("loading model!")
        self.model = load_model(self.save_model_path,
                                custom_objects={
                                    'GradientReversal': GradientReversal,
                                    'K': K
                                })
        return self.model

    def compile_model(self,
                      optimizer=OPTIMIZER,
                      metrics=None,
                      loss_weights=None):
        '''
		This function compiles the model based on the given optimization method and its parameters.
		'''

        if metrics is None:
            metrics = ['accuracy']
        if loss_weights is None:
            loss_weights = {
                'classifier_output_aux': 1,
                'classifier_output_main': 1,
            }
        self.model.compile(optimizer=optimizer,
                           metrics=metrics,
                           loss={
                               'classifier_output_aux':
                               'categorical_crossentropy',
                               'classifier_output_main':
                               'categorical_crossentropy',
                           },
                           loss_weights=loss_weights)
        return self.model

    def fit_model(self, x_train_1, x_train_2, y_train_aux, y_train_main):
        """Fit the model on a training set.
        
        The training sets must be divided in batches that contain both the source and target domain, 
        in order to perform adaptation
        Returns
        -------
        type : keras.History
            The history of the trained model as a keras.History object.
        """
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=self.n_patience)

        model_history = self.model.fit(x=[x_train_1, x_train_2],
                                       y=[y_train_aux, y_train_main],
                                       epochs=self.n_epochs,
                                       batch_size=self.n_batch_size,
                                       validation_split=0.1,
                                       verbose=2,
                                       callbacks=[early_stopping])

        return model_history

    def predict_proba(self, x_test_1, x_test_2, y_test_aux, y_test_main):
        '''
		This function evaluates the model, and generates the predicted classes.
		'''
        save_model_path = self.save_model_path
        begin_time = time.clock()
        scores = self.model.evaluate(x=[x_test_1, x_test_2],
                                     y=[y_test_aux, y_test_main],
                                     verbose=0)
        end_time = time.clock()
        print("time_consuming:",(end_time-begin_time),y_test_aux.shape[0],(end_time-begin_time)/y_test_aux.shape[0])
        [y_test_aux_prefict,
         y_test_main_prefict] = self.model.predict([x_test_1, x_test_2])

        y_test_aux_prefict = np.argmax(y_test_aux_prefict, axis=-1) + 1
        y_test_aux = np.argmax(y_test_aux, axis=-1) + 1

        y_test_main_prefict = np.argmax(y_test_main_prefict, axis=-1) + 1
        y_test_main = np.argmax(y_test_main, axis=-1) + 1

        cm_aux = confusion_matrix(y_test_aux, y_test_aux_prefict)
        print("cm_aux:", cm_aux)
        cm_aux = cm_aux.astype('float') / cm_aux.sum(axis=1)[:, np.newaxis]
        cm_main = confusion_matrix(y_test_main, y_test_main_prefict)
        print("cm_main:", cm_main)
        cm_main = cm_main.astype('float') / cm_main.sum(axis=1)[:, np.newaxis]
        print("normalized cm_main:", cm_main)

        print("metrics:", self.model.metrics_names)
        print('Test score:', scores)
        print('Test loss:', scores[1])
        print('Test accuracy of aux:', scores[3])
        print('Test accuracy of main:', scores[4])

        # Save model to file
        print("Saving model to disk \n")
        self.model.save(save_model_path)

        return [cm_main, scores]
