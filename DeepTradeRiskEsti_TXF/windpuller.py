# -*- coding: utf-8 -*-
# Copyright 2017 The Xiaoyu Fang. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from keras.layers import Dense, LSTM, Activation, BatchNormalization, Dropout, initializers
from renormalization import BatchRenormalization
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop , Nadam
from keras.models import load_model
from keras.initializers import Constant
from keras.layers import Input, Dense, LSTM, merge
from keras.models import Model
from keras import regularizers
from numpy import newaxis
import numpy as np

# input_shape = [30,40] # loss='risk_estimation'
class WindPuller(object): 
    def __init__(self, input_shape , modelType=0 , lr=0.01, n_layers=2, n_hidden=8, rate_dropout=0.2, loss='risk_estimation'): 
        
        print("initializing..., learing rate %s, n_layers %s, n_hidden %s, dropout rate %s." %(lr, n_layers, n_hidden, rate_dropout))
        
        '''
            Orig Model from deep trader keras
        '''
        if(modelType==0): 
            self.model = Sequential()
            # issue : maybe don't drop out input data
            self.model.add(Dropout(rate=0.2, input_shape=(input_shape[0], input_shape[1])))
            #self.model.add(Input(shape=(input_shape[0], input_shape[1]),name='lstm_imput' ) )
            
            for i in range(0, n_layers - 1):
                self.model.add(LSTM(n_hidden * 4, return_sequences=True, activation='tanh',
                                    recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform',
                                    recurrent_initializer='orthogonal', bias_initializer='zeros',
                                    dropout=rate_dropout, recurrent_dropout=rate_dropout))
            self.model.add(LSTM(n_hidden, return_sequences=False, activation='tanh',
                                    recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform',
                                    recurrent_initializer='orthogonal', bias_initializer='zeros',
                                    dropout=rate_dropout, recurrent_dropout=rate_dropout))
            self.model.add(Dense(1, kernel_initializer=initializers.glorot_uniform(),activity_regularizer=regularizers.l2(0.01)))
            # self.model.add(BatchNormalization(axis=-1, moving_mean_initializer=Constant(value=0.5),
            #               moving_variance_initializer=Constant(value=0.25)))
            self.model.add(BatchRenormalization(axis=-1, beta_init=Constant(value=0.5)))
            self.model.add(Activation('relu_limited')) #relu_limited
            #opt = RMSprop(lr=lr)
            opt = Nadam(lr=lr)
            self.model.compile(loss=loss,optimizer=opt,metrics=['accuracy']) 
            
        # Model Copied From Type 0 But Output negative Signal Only
        elif(modelType==1): #
            self.model = Sequential()
            # issue : maybe don't drop out input data
            self.model.add(Dropout(rate=0.2, input_shape=(input_shape[0], input_shape[1])))
            for i in range(0, n_layers - 1):
                self.model.add(LSTM(n_hidden * 4, return_sequences=True, activation='tanh',
                                    recurrent_activation='tanh', kernel_initializer='glorot_uniform',
                                    recurrent_initializer='orthogonal', bias_initializer='zeros',
                                    dropout=rate_dropout, recurrent_dropout=rate_dropout))
            self.model.add(LSTM(n_hidden, return_sequences=False, activation='tanh',
                                    recurrent_activation='tanh', kernel_initializer='glorot_uniform',
                                    recurrent_initializer='orthogonal', bias_initializer='zeros',
                                    dropout=rate_dropout, recurrent_dropout=rate_dropout))
            self.model.add(Dense(1, kernel_initializer=initializers.glorot_uniform(),activity_regularizer=regularizers.l2(0.01)))
            # self.model.add(BatchNormalization(axis=-1, moving_mean_initializer=Constant(value=0.5),
            #               moving_variance_initializer=Constant(value=0.25)))
            self.model.add(BatchRenormalization(axis=-1, beta_init=Constant(value=0.5)))
            self.model.add(Activation('relu_inverse'))
            #opt = RMSprop(lr=lr)
            opt = Nadam(lr=lr)
            self.model.compile(loss=loss,optimizer=opt,metrics=['accuracy']) 
    
        elif(modelType==2): # model for only output signal to predict positive/negative
            import tensorflow as tf
            def atan(x): 
                return tf.atan(x)
            
            self.model = Sequential()
            self.model.add(LSTM(input_dim=input_shape[0] ,output_dim=input_shape[1], return_sequences=True ,activation=atan)) # ,output_dim=input_shape[1]
            self.model.add(Dropout(0.2))
            self.model.add(BatchNormalization())
            self.model.add(Dense( output_dim = int(input_shape[1]/2),activity_regularizer=regularizers.l2(0.01)))
            self.model.add(Activation(atan))
            self.model.add(Dense( output_dim = 1,activity_regularizer=regularizers.l2(0.01)))
            self.model.add(Activation(atan))
            self.model.compile(optimizer='adam', loss='mse', metrics=['mse'])
            
        elif(modelType==3): # model used by raj and add some experiments
            
            self.model = Sequential()
            self.model.add(LSTM(input_dim=input_shape[0],output_dim=input_shape[1],return_sequences=True))
            self.model.add(Dropout(0.2))
            self.model.add(BatchNormalization())
            
            self.model.add(LSTM(input_shape[2],return_sequences=False))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(output_dim=input_shape[3],activity_regularizer=regularizers.l2(0.01)))
            self.model.add(Activation("linear"))
            opt = Nadam(lr=lr)
            #self.model.compile(optimizer=opt , loss='risk_estimation', metrics=['accuracy'])
            self.model.compile(loss="mse", optimizer="rmsprop")
            
            
    def fit(self, x, y, batch_size=32, nb_epoch=100, verbose=1, callbacks=None,
            validation_split=0., validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None, initial_epoch=0, **kwargs):
        self.model.fit(x, y, batch_size, nb_epoch, verbose, callbacks,
                       validation_split, validation_data, shuffle, class_weight, sample_weight,
                       initial_epoch, **kwargs)

    def save(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = load_model(path)
        return self

    def evaluate(self, x, y, batch_size=32, verbose=1,
                 sample_weight=None, **kwargs):
        return self.model.evaluate(x, y, batch_size, verbose,
                            sample_weight, **kwargs)

    def predict(self, x, batch_size=32, verbose=0):
        return self.model.predict(x, batch_size, verbose)


    def predict_point_by_point(self, data):
        #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted
    
    def predict_sequence_full(self, data, window_size):
        #Shift the window by 1 new prediction each time, re-run predictions on new window
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        return predicted
    
    def predict_sequences_multiple(self, data, window_size, prediction_len):
        #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        prediction_seqs = []
        for i in range(int(len(data)/prediction_len)):
            curr_frame = data[i*prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs