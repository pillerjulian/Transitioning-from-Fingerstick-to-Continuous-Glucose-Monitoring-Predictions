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
    """
    Add artificial SMBG values to the data
    :param data: data to add artificial SMBG values to
    :param time_between_two_SMBGs_in_h: maximal time window between two SMBG values
    :return: data with artificial SMBG values and the indices of the artificial SMBG values
    """

    # Convert the maximal time window between two SMBG values into number of samples
    time_step_distance = round(time_between_two_SMBGs_in_h * 12)  # hours * 60min/hour / 5min/step
    artificial_SMBG_indices = [0]

    # Get indices of available SMBG values
    SMBG_indices = data['finger'][data['finger'].notna()].index
    # Calculate the distance between two successive available values and if the distance is longer
    # than the defined maximum, insert additional artificial SMBG values
    for i in range(len(SMBG_indices)):
        if i == 0:  # first value --> distance to this value
            distance = SMBG_indices[i]
        elif i == len(SMBG_indices) - 1:  # last value --> distance to the end
            distance = data['cbg'].index[-1] - SMBG_indices[i]
        else:  # not first or last index --> distance between two successive values
            distance = SMBG_indices[i + 1] - SMBG_indices[i]
        if distance > time_step_distance:
            num_artificial_points = math.ceil(
                distance / time_step_distance) - 1  # For example: twice the distance --> add one artificial point
            if num_artificial_points > 0:
                # For each needed atrificial SMBG value, calculate the index and copy
                # the CGM value at this index
                for j in range(num_artificial_points):
                    if i == 0:
                        artificial_index = time_step_distance * (j + 1)
                    elif i == len(SMBG_indices) - 1:
                        artificial_index = SMBG_indices[i] + time_step_distance * (j + 1)
                        if artificial_index > data['cbg'].index[
                            -1]:  # check if artificial index is outside of CGM signal
                            break
                    else:
                        artificial_index = SMBG_indices[i] + time_step_distance * (j + 1)
                    data['finger'].iloc[artificial_index] = data['cbg'].iloc[artificial_index]
                    # Save artificial indices for the differentiation between artificial and real
                    # SMBG value in the result figure
                    artificial_SMBG_indices.append(artificial_index)

    # Create a first SMBG at time 0 with the value of the first CGM value
    data['finger'].iloc[0] = data['cbg'].iloc[0]
    return data, artificial_SMBG_indices


def preprocess_data(data, SMBG_window):
    """
    Preprocess the data
    :param data: data to preprocess
    :param SMBG_window: SMBG window
    :return: preprocessed data and the indices of the artificial SMBG values
    """

    # Interpolate the zones with missing 'cbg' so there is not regions with super steep changes
    data['cbg'] = data['cbg'].interpolate(method='linear', limit_direction='both')

    # Replace any 0 values in 'finger' with corresponding values from 'cbg'
    data.loc[data['finger'] == 0, 'finger'] = data.loc[data['finger'] == 0, 'cbg']

    # Add artificial SMBG values
    data, artificial_SMBG_indices = add_artificial_SMBG(data, SMBG_window)

    # Periods of no food or bolus injected
    data['bolus'].fillna(0.0, inplace=True)
    data['carbInput'].fillna(0.0, inplace=True)
    # Periods of no basal insulin
    # data['basal'].fillna(0.0, inplace=True)  # TODO: Check if that's ok

    # data['finger'].fillna(0.0, inplace=True)

    # data['cbg'].fillna(0.0, inplace=True)

    return data, artificial_SMBG_indices


def read_patient(train_set, test_set, prediction_window, SMBG_window):
    """
    Read the data of a patient and preprocess it
    :param train_set: training set
    :param test_set: testing set
    :param prediction_window: prediction window
    :param SMBG_window: SMBG window
    :return: preprocessed data, scalers for the train and test set, and the indices of the artificial SMBG values
    """

    # Preprocess the data
    train_set, train_artificial_SMBG_indices = preprocess_data(train_set, SMBG_window)
    test_set, test_artificial_SMBG_indices = preprocess_data(test_set, SMBG_window)

    test_set_plotting = test_set  # save dataframe for the plotting of SMBG values in the results

    # Feature selection, cbg is for y_train, the rest are for x_train
    features = ['cbg', 'finger', 'basal', 'carbInput', 'bolus']
    train_set = train_set[features].values
    test_set = test_set[features].values

    # Scaling data from 0 to 1 for each individual feature
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
        scaled_column[isnan_mask] = 0  # we can set here either to -1 or 0, still have

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

    x_train = []  # holds the features (all except cbg) with window size equal the prediction window
    y_train = []  # holds the cbg with window size equal the prediction window

    # Append the features into x_train and cbg into y_train, considering prediction_window for each
    # i = prediction_window
    # while i < len(scaled_train_set):
    #     x_train.append(scaled_train_set[i - prediction_window:i, 1:])
    #     y_train.append(scaled_train_set[i - prediction_window:i, 0])
    #     i = i + prediction_window

    # # Splitting the vector where y_train is cbg and x_train has the other features
    for i in range(prediction_window, len(scaled_train_set)):
        x_train.append(scaled_train_set[i - prediction_window:i, 1:])
        y_train.append(scaled_train_set[i - prediction_window:i,
                       0])  # to predict, 0 corresponds to 'cbg' as the prediction target ##### maybe wrong

    x_train, y_train = np.array(x_train), np.array(y_train)  # converting from list to numpy array

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], len(features) - 1))  # number of windows,
    # window size, features except cbg

    # print(x_train.shape) (11531, 80, 4)
    # print(y_train.shape) (11531, 80)

    x_test = []  # arrays of blood glucose with len of input_len
    y_test = []  # arrays of blood glucose with len of output_len
    ground_truth_cbg = []  # creates a 1D ground truth vector of cgm, used for plotting only

    i = prediction_window
    while i < len(scaled_test_set):
        x_test.append(scaled_test_set[i - prediction_window:i, 1:])
        y_test.append(test_set[i - prediction_window:i, 0])
        for j in test_set[i - prediction_window:i, 0]:
            ground_truth_cbg.append(j)
        i = i + prediction_window  # jump output_len values in the future

    x_test = np.array(x_test)  # converting to numpy array
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], len(features) - 1))

    # print(x_test.shape) (35, 80, 4)
    # print(np.array(y_test).shape) (35, 80)

    return x_train, y_train, x_test, y_test, ground_truth_cbg, scalers_transforms_train, scalers_transforms_test, test_set_plotting, train_artificial_SMBG_indices, test_artificial_SMBG_indices

def train_model(x_train, y_train, batch_size, epochs, learning_rate):
    """
    Create and train the model
    :param x_train: training input data
    :param y_train: training target data
    :param batch_size: batch size
    :param epochs: number of epochs
    :param learning_rate: learning rate
    :return: the trained model and the history of the training
    """
    prediction_window_size = x_train.shape[1]
    num_features = x_train.shape[2]  # number of features except cbg (finger, basal, carbInput, bolus)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(prediction_window_size, num_features)))
    model.add(LSTM(units=50, return_sequences=False))  # try activation = 'relu' inside this 2 layers
    # model.add(Dropout(0.2))
    model.add(Dense(units=y_train.shape[1]))

    # custom_optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer="adam", loss='mse')
    # model.compile(optimizer=custom_optimizer, loss='mse')

    # TODO: ADAM uses learning_rate=0.001 as default, we can try different values
    # TODO: try with Relu activation function
    # TODO: try more LSTM Units
    # TODO: Try more/less dropout

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    return model, history


def make_prediction(scalers_transforms_test, model, x_test, y_test):
    """
    Make predictions on the test set
    :param scalers_transforms_test:
    :param model:
    :param x_test:
    :param y_test:
    :return:
    """
    # Make predictions
    predictions = model.predict(x_test)
    predictions = np.reshape(predictions, (predictions.shape[0], predictions.shape[1]))  # reshape to match y_test
    predictions = scalers_transforms_test[0].inverse_transform(predictions)  # bring values back to original scale

    # creates a 1D prediction vector of cgm, used for plotting only
    predicted_cbg = predictions[0]
    for i in range(1, len(predictions)):  # TODO: Check if its correct
        predicted_cbg = np.concatenate([predicted_cbg, predictions[i]])

    # print("shape predicted_cbg:", predicted_cbg.shape) # shape predicted_cbg: (2800,)

    # Calculate the RMSE
    y_test = np.array(y_test)

    print("shape y_test:", y_test.shape) # shape y_test: (35, 80)
    print("shape prediction:", predictions.shape) # shape prediction: (35, 80)

    rmse = round(np.sqrt(np.mean(((predictions - y_test) ** 2))), 4)

    return predictions, predicted_cbg, rmse


# def create_and_save_plots(continuous_ytest, continuous_predictions, test_set_plotting, rmse, history,
#                           batch_size, epochs, learning_rate, prediction_window,
#                           test_artificial_SMBG_indices, inter_statement):
#     # Create a directory for saving plots inside the 'results' folder
#     time_stamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#     dir_name = f"results/plots_{time_stamp}"  # Prepend 'results/' to the directory name
#     os.makedirs(dir_name, exist_ok=True)
#     # Plotting the predictions
#     plt.figure(figsize=(16, 8))
#     plt.title(f'Blood Glucose Prediction Model Result with RMSE: {rmse} ({inter_statement})')
#     # plt.plot(continuous_ytest, color='b')
#     plt.plot(np.arange(len(test_set_plotting['cbg'])), test_set_plotting['cbg'],color='blue')
#     plt.plot(continuous_predictions, color='r')
#     # plt.scatter(np.arange(len(smbg_scatter)), smbg_scatter, color='black', marker='o')
#     plt.plot(np.arange(len(test_set_plotting['finger'])), test_set_plotting['finger'], marker='o', linestyle='-',
#              color='black')  # plot all SMBG values (artificial and real)
#     plt.scatter(np.arange(len(test_set_plotting['finger']))[test_artificial_SMBG_indices],
#                 test_set_plotting['finger'].iloc[test_artificial_SMBG_indices], marker='o', color='green', s=50,
#                 zorder=5)   # plot the artifical SMBG values in another color
#     plt.xlabel('Timestamp', fontsize=18)
#     plt.ylabel('CGM (mg/dL)', fontsize=18)
#     plt.legend(['Real', 'Predictions', "SMBG", "Artificial SMBG"], loc='lower right')
#     plt.figtext(0.5, 0.01, f"Parameters: batch size: {batch_size}; #epochs: {epochs}; learning rate: {learning_rate};"
#                            f" #prediction window: {prediction_window}", ha='center')
#     plt.savefig(f"{dir_name}/test_{time_stamp}.png")
#     plt.close()  # Close the plot to free up memory
#
#     # Plotting the training loss
#     plt.figure(figsize=(12, 6))
#     plt.plot(history.history['loss'])
#     plt.title(f'Model Loss ({inter_statement})')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['Train'], loc='upper right')
#     plt.figtext(0.5, 0.01, f"Parameters: batch size: {batch_size}; #epochs: {epochs}; learning rate: {learning_rate};"
#                            f" #prediction window: {prediction_window}", ha='center')
#     plt.savefig(f"{dir_name}/loss_{time_stamp}.png")
#     plt.close()  # Close the plot to free up memory

def create_and_save_plots(continuous_ytest, continuous_predictions, test_set_plotting, rmse, history,
                          batch_size, epochs, learning_rate, prediction_window,
                          test_artificial_SMBG_indices, inter_statement):
    '''

    :param continuous_ytest:
    :param continuous_predictions:
    :param test_set_plotting:
    :param rmse:
    :param history:
    :param batch_size:
    :param epochs:
    :param learning_rate:
    :param prediction_window:
    :param test_artificial_SMBG_indices:
    :param inter_statement:
    :return:
    '''
    # Create a directory for saving plots inside the 'results' folder
    time_stamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    dir_name = f"results/plots_{time_stamp}"  # Prepend 'results/' to the directory name
    os.makedirs(dir_name, exist_ok=True)
    # Plotting the predictions
    plt.figure(figsize=(16, 8))
    plt.title(f'Blood Glucose Prediction Model Result with RMSE: {rmse} ({inter_statement})')
    plt.plot(continuous_ytest, color='b')
    # plt.plot(np.arange(len(test_set_plotting['cbg'].iloc[:len(continuous_predictions)])), test_set_plotting['cbg'].iloc[:len(continuous_predictions)],color='blue')
    plt.plot(continuous_predictions, color='r')
    # plt.scatter(np.arange(len(smbg_scatter)), smbg_scatter, color='black', marker='o')

    plt.plot(np.arange(len(test_set_plotting['finger'].iloc[:len(continuous_predictions)])), test_set_plotting['finger'].iloc[:len(continuous_predictions)], marker='o', linestyle='-',
             color='black')  # plot all SMBG values (artificial and real)
    test_artificial_SMBG_indices = np.array(test_artificial_SMBG_indices)
    test_artificial_SMBG_indices = test_artificial_SMBG_indices[test_artificial_SMBG_indices < len(continuous_predictions)]
    plt.scatter(np.arange(len(test_set_plotting['finger']))[test_artificial_SMBG_indices],
                test_set_plotting['finger'].iloc[test_artificial_SMBG_indices], marker='o', color='green', s=50,
                zorder=5)   # plot the artifical SMBG values in another color
    plt.xlabel('Timestamp', fontsize=18)
    plt.ylabel('CGM (mg/dL)', fontsize=18)
    plt.legend(['Real', 'Predictions', "SMBG", "Artificial SMBG"], loc='lower right')
    plt.figtext(0.5, 0.01, f"Parameters: batch size: {batch_size}; #epochs: {epochs}; learning rate: {learning_rate};"
                           f" #prediction window: {prediction_window}", ha='center')
    plt.savefig(f"{dir_name}/test_{time_stamp}.png")
    plt.close()  # Close the plot to free up memory

    # Plotting the training loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'])
    plt.title(f'Model Loss ({inter_statement})')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper right')
    plt.figtext(0.5, 0.01, f"Parameters: batch size: {batch_size}; #epochs: {epochs}; learning rate: {learning_rate};"
                           f" #prediction window: {round(prediction_window*5/60, 2)} hours", ha='center')
    plt.savefig(f"{dir_name}/loss_{time_stamp}.png")
    plt.close()  # Close the plot to free up memory


if __name__ == "__main__":
    '''
    Main function
    
    '''

    # Path to the data folder

    # Get the current working directory
    current_directory = os.getcwd()
    # print(current_directory)
    data_folder = os.path.join(current_directory, 'dataset')

    # UNCOMMENT THIS PART IF USED ON LOCAL MACHINE
    # Go one step up from the current directory
    # parent_directory = os.path.dirname(current_directory)
    # data_folder = os.path.join(parent_directory, 'dataset')

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
    inter_model = True
    inter_statement = "intra-patient model"  # Dont change
    patient_index = 2
    all_train_inter_model = []
    data_from_2018 = True   # if true, it uses 2020 and 2018 set together for the training of inter model
    if inter_model:  # Concatenate the data of all patients excluding the patient under investigation
        inter_statement = "inter-patient model"
        for i in range(len(patient_ids_2018)):
            if i != patient_index:
                all_train_inter_model.append(
                    train_set_2018[i])
                all_train_inter_model.append(
                    train_set_2020[i])
                # pd.concat([all_train_inter_model, train_set_2018[i]], ignore_index=
            else:
                if data_from_2018:
                    all_train_inter_model.append(
                        train_set_2020[i])
                else:
                    all_train_inter_model.append(
                        train_set_2018[i])
        all_train_inter_model = pd.concat(all_train_inter_model, ignore_index=True)
        train_input_data = all_train_inter_model
    else:
        train_input_data = train_set_2018[patient_index]  # only one patient

    # Define Model Hyperparameters: (ADAPT ONLY HERE THE PARAMETERS)
    batch_size = 30
    epochs = 2
    learning_rate = 0.001
    prediction_window = 50
    SMBG_window = 4  # in hours
    x_train, y_train, x_test, y_test, continuous_ytest, scalers_transforms_train, scalers_transforms_test, test_set_plotting, train_artificial_SMBG_indices, test_artificial_SMBG_indices = read_patient(
        train_input_data, test_set, prediction_window, SMBG_window)
    model, history = train_model(x_train, y_train, batch_size, epochs, learning_rate)
    predictions, continuous_predictions, rmse = make_prediction(scalers_transforms_test, model, x_test, y_test)
    # show_plots(continuous_ytest, continuous_predictions,  smbg_scatter, rmse)
    # create_history_plot(history)
    create_and_save_plots(continuous_ytest, continuous_predictions, test_set_plotting, rmse, history,
                          batch_size, epochs, learning_rate, prediction_window,
                          test_artificial_SMBG_indices, inter_statement)
