import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os # Import the os module for path manipulation
import tensorflow as tf # Import tensorflow to save/load models

# Function to parse XML data
def parse_glucose_data(xml_file_path):
    """
    Parses an XML file containing glucose level data and returns a pandas DataFrame.

    Args:
        xml_file_path (str): The path to the XML file.

    Returns:
        pd.DataFrame: A DataFrame with 'timestamp' and 'glucose_value' columns.
    """
    # Print debugging information about paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script is running from directory: {script_dir}")
    absolute_file_path = os.path.abspath(os.path.join(script_dir, xml_file_path))
    print(f"Attempting to open file at absolute path: {absolute_file_path}")

    try:
        tree = ET.parse(absolute_file_path) # Use the absolute path for parsing
        root = tree.getroot()

        data = []
        # Find all 'event' tags within 'glucose_level'
        for event in root.findall('.//glucose_level/event'):
            ts = event.get('ts')
            value = int(event.get('value'))
            data.append({'timestamp': ts, 'glucose_value': value})

        df = pd.DataFrame(data)
        # Convert timestamp string to datetime objects
        # FIX: Changed format from %m-%d-%Y to %d-%m-%Y based on error message
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M:%S')
        # Sort data by timestamp to ensure correct time series order
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        return df
    except FileNotFoundError:
        print(f"Error: The file '{absolute_file_path}' was not found. Please check the path.")
        print(f"Current working directory (from os.getcwd()): {os.getcwd()}")
        return pd.DataFrame() # Return empty DataFrame on error
    except ET.ParseError as e:
        print(f"Error parsing XML file '{absolute_file_path}': {e}")
        return pd.DataFrame()
    except ValueError as e: # Catch the specific ValueError for date format issues
        print(f"Error converting timestamp in '{absolute_file_path}': {e}")
        print("Please check the date format in your XML file. Expected format: DD-MM-YYYY HH:MM:SS")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred while parsing '{absolute_file_path}': {e}")
        return pd.DataFrame()


# Function to create sequences for LSTM
def create_sequences(data, look_back=1):
    """
    Creates sequences for LSTM training.
    For a given look_back window, it creates input sequences (X)
    and corresponding target values (y).

    Args:
        data (np.array): The input data (e.g., normalized glucose values).
                         Expected shape: (n_samples, n_features).
        look_back (int): The number of previous time steps to use as input features
                         to predict the next time step.

    Returns:
        tuple: A tuple containing (X, y) where X are the input sequences
               and y are the targets.
    """
    X, y = [], []
    for i in range(len(data) - look_back):
        # X will contain 'look_back' previous glucose values
        X.append(data[i:(i + look_back), 0])
        # y will contain the glucose value immediately following the sequence in X
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

def save_statistics_to_file(metrics_dict, file_path, plot_paths=None):
    """
    Saves a dictionary of evaluation metrics to a text file,
    and optionally includes paths to generated plots.

    Args:
        metrics_dict (dict): A dictionary where keys are metric names (str)
                             and values are their corresponding scores (float).
        file_path (str): The path to the text file where statistics will be saved.
        plot_paths (dict, optional): A dictionary of plot names and their file paths.
                                     E.g., {"Training Loss Plot": "path/to/loss.png"}.
    """
    try:
        with open(file_path, 'w') as f:
            f.write("--- Model Performance Statistics ---\n\n")
            for key, value in metrics_dict.items():
                f.write(f"{key}: {value:.4f}\n") # Format to 4 decimal places for consistency

            if plot_paths:
                f.write("\n--- Associated Plots ---\n")
                for plot_name, plot_path in plot_paths.items():
                    f.write(f"{plot_name}: {os.path.abspath(plot_path)}\n") # Write absolute path

        print(f"\nStatistics saved successfully to {file_path}")
    except Exception as e:
        print(f"Error saving statistics to file {file_path}: {e}")

# Main script logic to train the LSTM model
def train_glucose_lstm(train_xml_file_path, test_xml_file_path, look_back=10, epochs=50, batch_size=32, model_save_path='glucose_lstm_model.h5', stats_save_path='glucose_prediction_statistics.txt', loss_plot_path='training_loss.png', prediction_plot_path='test_predictions.png'):
    """
    Trains an LSTM model for glucose level prediction and evaluates it on a separate test set.
    The trained model and scaler are saved, and performance statistics and plot paths are saved to a file.

    Args:
        train_xml_file_path (str): Path to the XML file containing training glucose data.
        test_xml_file_path (str): Path to the XML file containing testing glucose data.
        look_back (int): Number of previous time steps (glucose readings) to use
                         as input for predicting the next reading.
        epochs (int): Number of training epochs (iterations over the entire dataset).
        batch_size (int): Number of samples per gradient update during training.
        model_save_path (str): File path to save the trained Keras model.
        stats_save_path (str): File path to save the performance statistics.
        loss_plot_path (str): File path to save the training loss plot.
        prediction_plot_path (str): File path to save the test predictions plot.
    """
    print(f"--- Training Data Processing ---")
    print(f"Parsing training data from {train_xml_file_path}...")
    df_train = parse_glucose_data(train_xml_file_path)

    if df_train.empty:
        print("Training data parsing failed or no data found. Exiting training.")
        return None, None # Return None for model and scaler if training data is bad

    print("Training data parsed successfully.")
    print(f"Total training data points: {len(df_train)}")

    # Normalize the training glucose values
    scaler = MinMaxScaler(feature_range=(0, 1))
    glucose_values_train = df_train['glucose_value'].values.reshape(-1, 1)
    scaled_glucose_values_train = scaler.fit_transform(glucose_values_train)

    # Create sequences for training
    X_train, y_train = create_sequences(scaled_glucose_values_train, look_back)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    print(f"Training data input shape: {X_train.shape}, target shape: {y_train.shape}")

    # Build the LSTM model
    print("\n--- Building LSTM Model ---")
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print("Model summary:")
    model.summary()

    # Train the model
    print("\n--- Training Model ---")
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        verbose=1, shuffle=False)
    print("Model training complete.")

    # Save the trained model
    try:
        model.save(model_save_path)
        print(f"\nModel saved successfully to {model_save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

    # Plot training history (loss over epochs) and save it
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.title('Model Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Mean Squared Error)')
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_plot_path) # Save the plot
    plt.close() # Close the plot to prevent it from displaying immediately
    print(f"Training loss plot saved to {loss_plot_path}")


    print(f"\n--- Evaluation on Training Data ---")
    train_predict = model.predict(X_train)
    train_predict_inv = scaler.inverse_transform(train_predict)
    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))

    train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_predict_inv))
    train_mae = mean_absolute_error(y_train_inv, train_predict_inv)
    train_r2 = r2_score(y_train_inv, train_predict_inv)

    print(f"Training RMSE: {train_rmse:.2f}")
    print(f"Training MAE: {train_mae:.2f}")
    print(f"Training R-squared: {train_r2:.2f}")

    # Prepare metrics for saving
    metrics_to_save = {
        "Training RMSE": train_rmse,
        "Training MAE": train_mae,
        "Training R-squared": train_r2
    }

    # Prepare plot paths for saving
    plot_paths_to_save = {
        "Training Loss Plot": loss_plot_path
    }


    print(f"\n--- Testing Data Processing & Evaluation ---")
    print(f"Parsing testing data from {test_xml_file_path}...")
    df_test = parse_glucose_data(test_xml_file_path)

    if df_test.empty:
        print("Testing data parsing failed or no data found. Cannot evaluate on test set.")
        # Save only training metrics and loss plot path if test data is missing
        save_statistics_to_file(metrics_to_save, stats_save_path, plot_paths_to_save)
        return model, scaler # Return model and scaler even if test data is bad

    print("Testing data parsed successfully.")
    print(f"Total testing data points: {len(df_test)}")

    # Scale the test glucose values using the SAME SCALER fitted on training data
    glucose_values_test = df_test['glucose_value'].values.reshape(-1, 1)
    scaled_glucose_values_test = scaler.transform(glucose_values_test)

    # Create sequences for testing
    X_test, y_test = create_sequences(scaled_glucose_values_test, look_back)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    print(f"Testing data input shape: {X_test.shape}, target shape: {y_test.shape}")

    # Make predictions on the test set
    test_predict = model.predict(X_test)
    test_predict_inv = scaler.inverse_transform(test_predict)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate evaluation metrics for the test set
    test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_predict_inv))
    test_mae = mean_absolute_error(y_test_inv, test_predict_inv)
    test_r2 = r2_score(y_test_inv, test_predict_inv)

    print(f"\nTest RMSE: {test_rmse:.2f}")
    print(f"Test MAE: {test_mae:.2f}")
    print(f"Test R-squared: {test_r2:.2f}")

    # Add test metrics to the dictionary
    metrics_to_save["Test RMSE"] = test_rmse
    metrics_to_save["Test MAE"] = test_mae
    metrics_to_save["Test R-squared"] = test_r2

    # Plot actual vs predicted for the test set and save it
    plt.figure(figsize=(15, 7))
    plt.plot(y_test_inv, label='Actual Glucose Level (Test Set)')
    plt.plot(test_predict_inv, label='Predicted Glucose Level (Test Set)')
    plt.title('Glucose Level Prediction (Test Set)')
    plt.xlabel('Time Step')
    plt.ylabel('Glucose Level')
    plt.legend()
    plt.grid(True)
    plt.savefig(prediction_plot_path) # Save the plot
    plt.close() # Close the plot
    print(f"Test prediction plot saved to {prediction_plot_path}")

    # Add prediction plot path to the dictionary
    plot_paths_to_save["Test Predictions Plot"] = prediction_plot_path

    # Save all statistics and plot paths to file
    save_statistics_to_file(metrics_to_save, stats_save_path, plot_paths_to_save)

    print("\nTraining and evaluation complete.")
    return model, scaler # Return model and scaler for potential further use

# Example usage:
if __name__ == "__main__":
    # Define parameters at the top level so they are accessible
    LOOK_BACK = 10
    EPOCHS = 100
    BATCH_SIZE = 32

    # Define the relative paths to your training and testing XML files
    # These paths are relative to the directory where LSTM_Model.py is located.
    base_data_path = os.path.join('..', '..', 'CleanData', 'OhioT1DM')
    training_xml_path = os.path.join(base_data_path, '591-ws-training.xml')
    testing_xml_path = os.path.join(base_data_path, '591-ws-testing.xml')

    # Define the output folder name
    output_folder_name = 'patient-591'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder_path = os.path.join(script_dir, output_folder_name)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)
    print(f"Ensured output folder exists at: {output_folder_path}")

    # Define where the model and statistics will be saved, now inside the new folder
    model_save_filename = 'glucose_lstm_model_591.h5'
    stats_save_filename = 'glucose_prediction_statistics_591.txt'
    loss_plot_filename = 'glucose_training_loss_591.png'
    prediction_plot_filename = 'glucose_test_predictions_591.png'

    model_save_path = os.path.join(output_folder_path, model_save_filename)
    stats_save_path = os.path.join(output_folder_path, stats_save_filename)
    loss_plot_path = os.path.join(output_folder_path, loss_plot_filename)
    prediction_plot_path = os.path.join(output_folder_path, prediction_plot_filename)


    # Call the training function with your actual XML file paths
    trained_model, data_scaler = train_glucose_lstm(
        train_xml_file_path=training_xml_path,
        test_xml_file_path=testing_xml_path,
        look_back=LOOK_BACK,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        model_save_path=model_save_path,
        stats_save_path=stats_save_path,
        loss_plot_path=loss_plot_path,
        prediction_plot_path=prediction_plot_path
    )

    if trained_model and data_scaler:
        print("\n--- Demonstrating Model Loading and Prediction ---")
        try:
            loaded_model = load_model(model_save_path)
            print(f"Model loaded successfully from {model_save_path}")

            df_test_for_prediction = parse_glucose_data(testing_xml_path)
            if not df_test_for_prediction.empty and len(df_test_for_prediction) >= LOOK_BACK:
                last_sequence_raw = df_test_for_prediction['glucose_value'].values[-LOOK_BACK:].reshape(-1, 1)
                scaled_last_sequence = data_scaler.transform(last_sequence_raw)
                scaled_last_sequence = np.reshape(scaled_last_sequence, (1, LOOK_BACK, 1))

                predicted_scaled_value = loaded_model.predict(scaled_last_sequence)
                predicted_glucose_value = data_scaler.inverse_transform(predicted_scaled_value)
                print(f"\nPredicted next glucose level using loaded model (based on last {LOOK_BACK} test values): {predicted_glucose_value[0][0]:.2f}")
            else:
                print(f"Not enough data points ({len(df_test_for_prediction)}) in the test DataFrame to make a prediction using the defined look_back ({LOOK_BACK}).")

        except Exception as e:
            print(f"Error loading model or making prediction: {e}")
    else:
        print("\nModel training or data parsing failed, skipping model loading demonstration.")
