import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras as keras
import tensorflow as tf
import keras.backend as K

from datetime import datetime
from statsmodels.tools.eval_measures import rmse
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, TimeDistributed, Masking
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import warnings
import io
import math
warnings.filterwarnings("ignore")


def preprocess_data(data):
    # Replace any 0 values in 'finger' with corresponding values from 'cbg'
    data.loc[data['finger'] == 0, 'finger'] = data.loc[data['finger'] == 0, 'cbg'] 

    data['finger'][::50] =  data['cbg'][::50] 
    
    # Periods of no food or bolus injected
    data['bolus'].fillna(0.0, inplace=True)
    data['carbInput'].fillna(0.0, inplace=True)
    
    # Interpolate the zones with missing 'cbg' so there is not regions with super steep changes 
    # TODO: Only do interpolation when missing_cbg is small 
    # TODO: After doing interpolation, set the missing_cbg of the interpolated to 0
    data['cbg'] = data['cbg'].interpolate(method='linear', limit_direction='both')
    
    # BASAL, do we do anything there? there is periods with 0.0 and NaN???
    #  Basal is the rate at which basal insulin is continuously infused. If it NaN it should be 0 then no?
    
    return data 
	
def read_patient(train_set, test_set, finger_window, prediction_window):

    train_set = preprocess_data(train_set)
    test_set = preprocess_data(test_set)
    
    # Use cbg, smbg, bolus, carbInput and the other stuff as inputs
    features = ['cbg', 'finger', 'basal', 'hr', 'gsr', 'carbInput', 'bolus']
    train_set = train_set[features].values
    test_set = test_set[features].values

    # Scalling data from 0 - 1 for each individual feature, considering NaN to be unchanged 
    scaled_train_set = np.zeros((train_set.shape[0], train_set.shape[1]))
    scaled_test_set = np.zeros((test_set.shape[0], test_set.shape[1]))

    scalers_transforms_train = []
    scalers_transforms_test = []

    # Loop for the train set
    for i in range(train_set.shape[1]):
        # Extract the column and remember where NaNs are
        column = train_set[:, i]
        isnan_mask = np.isnan(column)

        # Replace NaN values with the mean of the non-NaN elements for scaling
        column_mean = np.nanmean(column)
        column[isnan_mask] = column_mean
    
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_column = scaler.fit_transform(column.reshape(-1, 1))
    
        # We store a -1 value where there is a NaN value 
        scaled_column[isnan_mask] = 0  # we can set here either to -1 or 0, still have # TODO
        
        scaled_train_set[:, i] = scaled_column.flatten()

        scalers_transforms_train.append(scaler)

        column[isnan_mask] = np.nan
        
    # Loop for the test set
    for i in range(test_set.shape[1]):
        # Extract the column and remember where NaNs are
        column = test_set[:, i]
        isnan_mask = np.isnan(column)

        # Replace NaN values with the mean of the non-NaN elements for scaling
        column_mean = np.nanmean(column)
        column[isnan_mask] = column_mean
    
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_column = scaler.fit_transform(column.reshape(-1, 1))
    
        # We store a -1 value where there is a NaN value 
        scaled_column[isnan_mask] = 0  # we can set here either to -1 or 0, still have # TODO
        
        scaled_test_set[:, i] = scaled_column.flatten()

        scalers_transforms_test.append(scaler)

        column[isnan_mask] = np.nan
            
    # train_data = scaled_data[0:training_data_len, : ] # (9289, 7)
    
    x_train=[] # arrays of blood glucose with len of input_len
    y_train = [] # arrays of blood glucose with len of output_len

    # 'cbg' is the target, but we're using all features for input
    for i in range(prediction_window,len(scaled_train_set)):
        x_train.append(scaled_train_set[i-prediction_window:i:finger_window, 1:]) # to learn past data 6 features 
        y_train.append(scaled_train_set[i-prediction_window:i, 0])  # TODO: Should we also give the SMBG values to
                                                                    # the model so it predicts the values in-between?
            
    x_train, y_train = np.array(x_train), np.array(y_train) # converting from list to numpy array
    
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], len(features)-1)) # windows, past values, features

    #############################################
    
    x_test = [] # arrays of blood glucose with len of input_len
    y_test = [] # arrays of blood glucose with len of output_len
    continuous_ytest = [] # list with not scaled blood glucose from y_test not broken into arrays


    i = prediction_window
    while (i >= prediction_window and i < len(scaled_test_set)):
        x_test.append(scaled_test_set[i-prediction_window:i:finger_window, 1:])
        y_test.append(test_set[i-prediction_window:i, 0]) ### maybe wrong
        for bg in test_set[i-prediction_window:i,0]:
            continuous_ytest.append(bg) # not for testing, just for plot purpose
        i = i+prediction_window # jump output_len values in the future
        
    x_test = np.array(x_test) # converting to numpy array
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],len(features)-1))

    smbg_scatter = test_set[:, 1]

    return x_train, y_train, x_test, y_test, continuous_ytest, scalers_transforms_train, scalers_transforms_test, smbg_scatter	
	
	
def rmse(x_train, y_train):
    return K.sqrt(K.mean(K.square(y_train - x_train)))

def train_model(x_train, y_train, batch_size, epochs, learning_rate):
    # LSTM Model 

    # Using the number of features in x_train to dynamically determine the input shape
    num_features = x_train.shape[2]
    
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], num_features))) #add batch normalization 
    model.add(LSTM(units=50, return_sequences=False)) # Return sequences for the next LSTM layer
    model.add(Dropout(0.5))
    model.add(Dense(units=y_train.shape[1]))
    # model.add(LSTM(units=50, return_sequences=True)) # Additional LSTM layer that returns sequences
    # model.add(TimeDistributed(Dense(1)))  # 1 or 7 ? because y_train has a shape of [?, 49, 7]

    custom_optimizer = Adam(learning_rate=learning_rate)

    # model.compile(optimizer="adam", loss='mse',metrics=['accuracy'])
    model.compile(optimizer=custom_optimizer, loss='mse', metrics=[rmse]) # TODO: Try different loss functions
    # TODO: ADAM uses learning_rate=0.001 as default, we can try different values 

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    return model, history

def make_prediction(scalers_transforms_test, model, x_test, y_test):

    # Make predictions
    predictions = model.predict(x_test)

    # Maybe we should give somehow the SMGB values to the model so it predicts the values in-between


    predictions = np.reshape(predictions, (predictions.shape[0],predictions.shape[1])) # reshape just like y_test

    predictions = scalers_transforms_test[0].inverse_transform(predictions) #########
    
    # for i in range(predictions.shape[1]):
    #     column = predictions[0, i].reshape(-1, 1)
    #     predictions = scalers_transforms[i].inverse_transform(column).flatten()

    # Create a continuous data of predictions to plot with continuous_ytest
    continuous_predictions = predictions[0]

    for i in range(1,len(predictions)):
        continuous_predictions = np.concatenate([continuous_predictions,predictions[i]])

    # continuous_predictions = scalers_transforms[0].inverse_transform(continuous_predictions.reshape(-1, 1)).flatten()


    y_test = np.array(y_test)
    rmse=np.sqrt(np.mean(((predictions-y_test)**2)))

    return predictions, continuous_predictions, rmse 

# def show_plots(continuous_ytest, continuous_predictions, smbg_scatter, rmse):
#     # Create a directory for results
#     time_stamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#     dir_name = f"plots_{time_stamp}"
#     os.makedirs(dir_name, exist_ok=True)
#
#     # Plotting the predictions
#     plt.figure(figsize=(16,8))
#     plt.title(f'Blood Glucose Prediction Model Result with RMSE: {rmse}')
#     plt.plot(continuous_ytest, color = 'b')
#     plt.plot(continuous_predictions, color = 'r')
#     plt.scatter(np.arange(len(smbg_scatter)), smbg_scatter, color = 'black', marker='o')
#     plt.xlabel('Timestamp',fontsize=18)
#     plt.ylabel('BGBG (mg/dL)',fontsize=18)
#     plt.legend(['Real','Predictions'], loc='lower right')
#
#     # Save the plot in the new directory
#     plt.savefig(f"{dir_name}/test_{time_stamp}.png")


# def create_history_plot(history):
#     # Create a directory for saving plots
#     time_stamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#     dir_name = f"plots_{time_stamp}"
#     os.makedirs(dir_name, exist_ok=True)
#
#     # Plotting the training loss
#     plt.figure(figsize=(12, 6))
#     plt.plot(history.history['loss'])
#     plt.title('Model loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['Train'], loc='upper right')
#
#     # Save the plot in the new directory
#     plt.savefig(f"{dir_name}/loss_{time_stamp}.png")


def create_and_save_plots(continuous_ytest, continuous_predictions, smbg_scatter, rmse, history,
                          batch_size, epochs, learning_rate, finger_window, prediction_window):
    # Create a directory for saving plots inside the 'results' folder
    time_stamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    dir_name = f"results/plots_{time_stamp}"  # Prepend 'results/' to the directory name
    os.makedirs(dir_name, exist_ok=True)

    # Plotting the predictions
    plt.figure(figsize=(16,8))
    plt.title(f'Blood Glucose Prediction Model Result with RMSE: {rmse}')
    plt.plot(continuous_ytest, color='b')
    plt.plot(continuous_predictions, color='r')
    plt.scatter(np.arange(len(smbg_scatter)), smbg_scatter, color='black', marker='o')
    plt.xlabel('Timestamp', fontsize=18)
    plt.ylabel('CGM (mg/dL)', fontsize=18)
    plt.legend(['Real','Predictions'], loc='lower right')
    plt.figtext(0.5, 0.01, f"Parameters: batch size: {batch_size}; #epochs: {epochs}; learning rate: {learning_rate};"
                        f" #finger window: {finger_window}; #prediction window: {prediction_window}", ha='center')
    plt.savefig(f"{dir_name}/test_{time_stamp}.png")
    plt.close()  # Close the plot to free up memory

    # Plotting the training loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper right')
    plt.figtext(0.5, 0.01, f"Parameters: batch size: {batch_size}; #epochs: {epochs}; learning rate: {learning_rate};"
                        f" #finger window: {finger_window}; #prediction window: {prediction_window}", ha='center')
    plt.savefig(f"{dir_name}/loss_{time_stamp}.png")
    plt.close()  # Close the plot to free up memory


if __name__ == "__main__":

    # Path to the data folder

    # Get the current working directory
    current_directory = os.getcwd()
    print(current_directory)
    # Go one step up from the current directory
    parent_directory = os.path.dirname(current_directory)
    # print(parent_directory)
    data_folder = os.path.join(parent_directory, 'dataset')
    print(data_folder)

    #data_folder = r'C:\Ohio_Data'

    # List of patient excel IDs in the year folders
    patient_ids_2018 = [559, 563, 570, 575, 588, 591]
    patient_ids_2020 = [540, 544, 552, 567, 584, 596]

    train_set_2018 = []
    test_set_2018 = []
    train_set_2020 = []
    test_set_2020 = []

    # not super pretty but works:
    names_train_2018 = []
    names_test_2018 = []
    names_train_2020 = []
    names_test_2020 = []

    # For 2018 data
    for patient_id in patient_ids_2018:
        for folder_type in ['training', 'testing']:
            file_name = f"{patient_id}-ws-{folder_type}_processed.csv"
            if folder_type == 'training':
                names_train_2018.append(file_name)
            else:
                names_test_2018.append(file_name)
            file_path = os.path.join(data_folder, 'Ohio2018_processed', folder_type[:-3], file_name)
            if os.path.exists(file_path): 
                patient_data = pd.read_csv(file_path)
                train_set_2018.append(patient_data) if folder_type == 'training' else test_set_2018.append(patient_data)
            else:
                raise FileNotFoundError("The target directory does not exist.")

    # For 2020 data
    for patient_id in patient_ids_2020:
        for folder_type in ['training', 'testing']:
            file_name = f"{patient_id}-ws-{folder_type}_processed.csv"
            if folder_type == 'training':
                names_train_2020.append(file_name)
            else:
                names_test_2020.append(file_name)
            file_path = os.path.join(data_folder, 'Ohio2020_processed', folder_type[:-3], file_name)
            if os.path.exists(file_path): 
                patient_data = pd.read_csv(file_path)
                train_set_2020.append(patient_data) if folder_type == 'training' else test_set_2020.append(patient_data)
            else:
                raise FileNotFoundError("The target directory does not exist.")
	
    # To plot correlation of all patients together we need to concatenate the datasets of each patient together
    all_train_2018 = pd.concat(train_set_2018, ignore_index=True)
    all_test_2018 = pd.concat(test_set_2018, ignore_index=True)
    all_train_2020 = pd.concat(train_set_2020, ignore_index=True)
    all_test_2020 = pd.concat(test_set_2020, ignore_index=True)    
			
    index_patient_570 = 2
    train_set = train_set_2018[index_patient_570]
    test_set = test_set_2018[index_patient_570]
    # Define Model Hyperparameters: (ADAPT ONLY HERE THE PARAMETERS)
    batch_size = 25
    epochs = 4
    learning_rate = 0.001
    finger_window = 1
    prediction_window = 80
    x_train, y_train, x_test, y_test, continuous_ytest, scalers_transforms_train, scalers_transforms_test, smbg_scatter = read_patient(train_set, test_set, finger_window, prediction_window)
    model, history = train_model(x_train, y_train, batch_size, epochs, learning_rate)
    predictions, continuous_predictions, rmse = make_prediction(scalers_transforms_test, model, x_test, y_test)
    # show_plots(continuous_ytest, continuous_predictions,  smbg_scatter, rmse)
    # create_history_plot(history)
    create_and_save_plots(continuous_ytest, continuous_predictions, smbg_scatter, rmse, history,
                          batch_size, epochs, learning_rate, finger_window, prediction_window)
