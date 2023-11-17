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


def add_artificial_SMBG(data, time_between_two_SMBGs_in_h):
    time_step_distance = round(time_between_two_SMBGs_in_h * 12)  # hours * 60min/hour / 5min/step
    artificial_SMBG_indices = []
    SMB_indices = data['finger'][data['finger'].notna()].index
    for i in range(len(SMB_indices)):
        if i == 0:
            distance = SMB_indices[i]
        elif i == len(SMB_indices) - 1:
            distance = data['cbg'].index[-1] - SMB_indices[i]
        else:
            distance = SMB_indices[i + 1] - SMB_indices[i]
        if distance > time_step_distance:
            num_artificial_points = math.ceil(
                distance / time_step_distance) - 1  # TODO CHECK twice the distance --> add one artificial point
            if num_artificial_points > 0:
                for j in range(num_artificial_points):
                    if i == 0:
                        artificial_index = time_step_distance * (j + 1)
                    else:
                        artificial_index = SMB_indices[i] + time_step_distance * (j + 1)
                    data['finger'].iloc[artificial_index] = data['cbg'].iloc[artificial_index]
                    artificial_SMBG_indices.append(artificial_index)
    return data, artificial_SMBG_indices


def preprocess_data(data, time_window):
    # Replace any 0 values in 'finger' with corresponding values from 'cbg'
    data.loc[data['finger'] == 0, 'finger'] = data.loc[data['finger'] == 0, 'cbg']

    # Interpolate the zones with missing 'cbg' so there is not regions with super steep changes
    # TODO: Only do interpolation when missing_cbg is small
    # TODO: After doing interpolation, set the missing_cbg of the interpolated to 0
    data['cbg'] = data['cbg'].interpolate(method='linear', limit_direction='both')
    # data['cbg'] = data['cbg'].interpolate(method='spline', order=3, limit_direction='both') # NOT WORKING

    # data['finger'][::50] =  data['cbg'][::50]
    data, artificial_SMBG_indices = add_artificial_SMBG(data, time_window)  # 4h window

    # Periods of no food or bolus injected
    # data['bolus'].fillna(0.0, inplace=True)
    # data['carbInput'].fillna(0.0, inplace=True)

    # BASAL, do we do anything there? there is periods with 0.0 and NaN???
    #  Basal is the rate at which basal insulin is continuously infused. If it NaN it should be 0 then no?

    return data, artificial_SMBG_indices


def read_patient(train_set, test_set, finger_window, prediction_window, SMBG_window):
    train_set, train_artificial_SMBG_indices = preprocess_data(train_set, SMBG_window)
    test_set, test_artificial_SMBG_indices = preprocess_data(test_set, SMBG_window)

    test_set_plotting = test_set

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

    x_train = []  # arrays of blood glucose with len of input_len
    y_train = []  # arrays of blood glucose with len of output_len

    # 'cbg' is the target, but we're using all features for input
    for i in range(prediction_window, len(scaled_train_set)):
        x_train.append(scaled_train_set[i - prediction_window:i:finger_window, 1:])  # to learn past data 6 features
        y_train.append(scaled_train_set[i - prediction_window:i,
                       0])  # to predict, 0 corresponds to 'cbg' as the prediction target ##### maybe wrong

    x_train, y_train = np.array(x_train), np.array(y_train)  # converting from list to numpy array

    x_train = np.reshape(x_train,
                         (x_train.shape[0], x_train.shape[1], len(features) - 1))  # windows, past values, features

    x_test = []  # arrays of blood glucose with len of input_len
    y_test = []  # arrays of blood glucose with len of output_len
    continuous_ytest = []  # list with not scaled blood glucose from y_test not broken into arrays

    i = prediction_window
    while (i >= prediction_window and i < len(scaled_test_set)):
        x_test.append(scaled_test_set[i - prediction_window:i:finger_window, 1:])
        y_test.append(test_set[i - prediction_window:i, 0])  ### maybe wrong
        for j in test_set[i - prediction_window:i, 0]:
            continuous_ytest.append(j)  # not for testing, just for plot purpose
        i = i + prediction_window  # jump output_len values in the future

    x_test = np.array(x_test)  # converting to numpy array
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], len(features) - 1))

    return x_train, y_train, x_test, y_test, continuous_ytest, scalers_transforms_train, scalers_transforms_test, test_set_plotting, train_artificial_SMBG_indices, test_artificial_SMBG_indices


def rmse(y_true, y_pred):
    # Define a custom Root Mean Squared Error (RMSE) function as a metric
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


# def train_model(x_train, y_train, batch_size, epochs, learning_rate):
#     # Using the number of features in x_train to dynamically determine the input shape
#     num_features = x_train.shape[2]
#
#     model = Sequential()
#     model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], num_features)))
#     model.add(LSTM(units=50, return_sequences=False))
#     model.add(Dropout(0.5))
#     model.add(Dense(units=y_train.shape[1]))
#     # model.add(LSTM(units=50, return_sequences=True)) # Additional LSTM layer that returns sequences
#     # model.add(TimeDistributed(Dense(1)))  # 1 or 7 ? because y_train has a shape of [?, 49, 7]
#
#     custom_optimizer = Adam(learning_rate=learning_rate)
#
#     # model.compile(optimizer="adam", loss='mse',metrics=['accuracy'])
#     model.compile(optimizer=custom_optimizer, loss='mse', metrics=[rmse]) # TODO WHY LOSS = MSE and metric = RMSE?
#     # TODO: ADAM uses learning_rate=0.001 as default, we can try different values
#
#     history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
#
#     return model, history

def train_model(x_train, y_train, batch_size, epochs, learning_rate):
    # Determine the number of features in x_train to dynamically determine the input shape
    num_features = x_train.shape[2]

    # Define the model
    # TODO: Test different dropout values, number of LSTM layers, number of LSTM units, etc.
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], num_features)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(
        Dropout(0.2))  # Dropout layer to prevent overfitting, usually used when dataset is large and overfitting

    # Added a TimeDistributed Dense layer, which applies a Dense (fully connected) operation to each timestep
    # independently. Since we predict one 'cbg' value per timestep, the output dimension is set to 1
    model.add(TimeDistributed(Dense(1)))

    custom_optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=custom_optimizer, loss='mse', metrics=[rmse])

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

    return model, history


def make_prediction(scalers_transforms_test, model, x_test, y_test):
    # Make predictions
    predictions = model.predict(x_test)
    predictions = np.reshape(predictions, (predictions.shape[0], predictions.shape[1]))  # reshape to match y_test
    predictions = scalers_transforms_test[0].inverse_transform(predictions)  # bring values back to original scale

    # Initialize an array to hold the continuous sequence of predictions
    continuous_predictions = predictions[0]

    # Concatenate the predictions from each batch into a continuous sequence of predictions for plotting
    for i in range(1, len(predictions)):
        continuous_predictions = np.concatenate([continuous_predictions, predictions[i]])

    # Calculate the RMSE
    y_test = np.array(y_test)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

    return predictions, continuous_predictions, rmse


def create_and_save_plots(continuous_ytest, continuous_predictions, test_set_plotting, rmse, history,
                          batch_size, epochs, learning_rate, finger_window, prediction_window,
                          test_artificial_SMBG_indices):
    # Create a directory for saving plots inside the 'results' folder
    time_stamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    dir_name = f"results/plots_{time_stamp}"  # Prepend 'results/' to the directory name
    os.makedirs(dir_name, exist_ok=True)

    # Plotting the predictions
    plt.figure(figsize=(16, 8))
    plt.title(f'Blood Glucose Prediction Model Result with RMSE: {rmse}')
    plt.plot(continuous_ytest, color='b')
    plt.plot(continuous_predictions, color='r')
    # plt.scatter(np.arange(len(smbg_scatter)), smbg_scatter, color='black', marker='o')
    plt.plot(np.arange(len(test_set_plotting['finger'])), test_set_plotting['finger'], marker='o', linestyle='-',
             color='black',
             label='finger')
    plt.scatter(np.arange(len(test_set_plotting['finger']))[test_artificial_SMBG_indices],
                test_set_plotting['finger'].iloc[test_artificial_SMBG_indices], marker='o', color='green',
                label='artificial finger', s=50, zorder=5)
    plt.xlabel('Timestamp', fontsize=18)
    plt.ylabel('CGM (mg/dL)', fontsize=18)
    plt.legend(['Real', 'Predictions', "SMBG", "Artificial SMBG"], loc='lower right')
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
    # print(current_directory)
    data_folder = os.path.join(current_directory, 'dataset')

    # UNCOMMENT THIS PART IF USED ON LOCAL MACHINE
    # Go one step up from the current directory
    parent_directory = os.path.dirname(current_directory)
    data_folder = os.path.join(parent_directory, 'dataset')

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

    # Select input data
    train_input_data = None
    inter_model = False
    patient_index = 2
    all_train_inter_model = []
    if inter_model:
        for i in range(len(patient_ids_2018)):
            if i != patient_index:
                all_train_inter_model.append(
                    train_set_2018[i])  # pd.concat([all_train_inter_model, train_set_2018[i]], ignore_index=True)
        all_train_inter_model = pd.concat(all_train_inter_model, ignore_index=True)
        train_input_data = all_train_inter_model
    else:
        train_input_data = train_set_2018[patient_index]

    # Define Model Hyperparameters: (ADAPT ONLY HERE THE PARAMETERS)
    batch_size = 15
    epochs = 15
    learning_rate = 0.001
    finger_window = 1
    prediction_window = 80
    SMBG_window = 4  # in hours
    x_train, y_train, x_test, y_test, continuous_ytest, scalers_transforms_train, scalers_transforms_test, test_set_plotting, train_artificial_SMBG_indices, test_artificial_SMBG_indices = read_patient(
        train_input_data, test_set, finger_window, prediction_window, SMBG_window)
    model, history = train_model(x_train, y_train, batch_size, epochs, learning_rate)
    predictions, continuous_predictions, rmse = make_prediction(scalers_transforms_test, model, x_test, y_test)
    # show_plots(continuous_ytest, continuous_predictions,  smbg_scatter, rmse)
    # create_history_plot(history)
    create_and_save_plots(continuous_ytest, continuous_predictions, test_set_plotting, rmse, history,
                          batch_size, epochs, learning_rate, finger_window, prediction_window,
                          test_artificial_SMBG_indices)
