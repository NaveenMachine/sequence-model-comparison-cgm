import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import tensorflow as tf

def parse_glucose_data(xml_file_path):
    """Parses an XML file for glucose data, returns DataFrame."""
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        data = []
        for event in root.findall('.//glucose_level/event'):
            ts = event.get('ts')
            value = int(event.get('value'))
            data.append({'timestamp': ts, 'glucose_value': value})

        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M:%S')
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at '{xml_file_path}'.")
        return pd.DataFrame()
    except ET.ParseError as e:
        print(f"Error parsing XML file '{xml_file_path}': {e}")
        return pd.DataFrame()
    except ValueError as e:
        print(f"Error converting timestamp in '{xml_file_path}': {e}")
        print("Expected date format: DD-MM-YYYY HH:MM:SS")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred parsing '{xml_file_path}': {e}")
        return pd.DataFrame()

def create_sequences(data, look_back=1, prediction_steps_ahead=1):
    """
    Creates sequences for LSTM training, predicting 'prediction_steps_ahead'
    future steps based on 'look_back' past steps.
    """
    X, y = [], []
    max_index_needed = look_back + prediction_steps_ahead - 1

    if len(data) <= max_index_needed:
        print(f"Warning: Not enough data points ({len(data)}) for look_back ({look_back}) and prediction horizon ({prediction_steps_ahead} steps).")
        return np.array([]), np.array([])

    for i in range(len(data) - max_index_needed):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back + prediction_steps_ahead - 1, 0])
    return np.array(X), np.array(y)

def save_statistics_to_file(metrics_dict, file_path, plot_paths=None):
    """Saves evaluation metrics and plot paths to a text file."""
    try:
        with open(file_path, 'w') as f:
            f.write("--- Model Performance Statistics ---\n\n")
            for key, value in metrics_dict.items():
                f.write(f"{key}: {value:.4f}\n")

            if plot_paths:
                f.write("\n--- Associated Plots ---\n")
                for plot_name, plot_path in plot_paths.items():
                    f.write(f"{plot_name}: {os.path.abspath(plot_path)}\n")
        print(f"\nStatistics saved to {file_path}")
    except Exception as e:
        print(f"Error saving statistics to file {file_path}: {e}")

def train_glucose_lstm(train_xml_file_path, test_xml_file_path, look_back=10, prediction_steps_ahead=1, epochs=50, batch_size=32, model_save_path='glucose_lstm_model.h5', stats_save_path='glucose_prediction_statistics.txt', loss_plot_path='training_loss.png', prediction_plot_path='test_predictions.png'):
    """Trains and evaluates an LSTM model for glucose prediction."""
    print(f"--- Processing Training Data from {train_xml_file_path} ---")
    df_train = parse_glucose_data(train_xml_file_path)
    if df_train.empty:
        print("Training data invalid. Exiting training.")
        return None, None

    scaler = MinMaxScaler(feature_range=(0, 1))
    glucose_values_train = df_train['glucose_value'].values.reshape(-1, 1)
    scaled_glucose_values_train = scaler.fit_transform(glucose_values_train)

    X_train, y_train = create_sequences(scaled_glucose_values_train, look_back, prediction_steps_ahead)
    if X_train.size == 0 or y_train.size == 0:
        print("Training sequences could not be created. Exiting.")
        return None, None
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    print(f"Training input shape: {X_train.shape}, target shape: {y_train.shape}")

    print("\n--- Building LSTM Model ---")
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()

    print("\n--- Training Model ---")
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=False)
    print("Model training complete.")

    try:
        model.save(model_save_path)
        print(f"\nModel saved to {model_save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.title('Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Training loss plot saved to {loss_plot_path}")

    print(f"\n--- Evaluation on Training Data ---")
    train_predict = model.predict(X_train)
    train_predict_inv = scaler.inverse_transform(train_predict)
    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))

    train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_predict_inv))
    train_mae = mean_absolute_error(y_train_inv, train_predict_inv)
    train_r2 = r2_score(y_train_inv, train_predict_inv)

    print(f"Training RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}, R-squared: {train_r2:.2f}")

    metrics_to_save = {
        "Training RMSE": train_rmse, "Training MAE": train_mae, "Training R-squared": train_r2
    }
    plot_paths_to_save = {"Training Loss Plot": loss_plot_path}

    print(f"\n--- Processing & Evaluating on Test Data from {test_xml_file_path} ---")
    df_test = parse_glucose_data(test_xml_file_path)
    if df_test.empty:
        print("Testing data invalid. Skipping test evaluation.")
        save_statistics_to_file(metrics_to_save, stats_save_path, plot_paths_to_save)
        return model, scaler

    glucose_values_test = df_test['glucose_value'].values.reshape(-1, 1)
    scaled_glucose_values_test = scaler.transform(glucose_values_test)

    X_test, y_test = create_sequences(scaled_glucose_values_test, look_back, prediction_steps_ahead)
    if X_test.size == 0 or y_test.size == 0:
        print("Testing sequences could not be created. Skipping test evaluation.")
        save_statistics_to_file(metrics_to_save, stats_save_path, plot_paths_to_save)
        return model, scaler
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    print(f"Testing input shape: {X_test.shape}, target shape: {y_test.shape}")

    test_predict = model.predict(X_test)
    test_predict_inv = scaler.inverse_transform(test_predict)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_predict_inv))
    test_mae = mean_absolute_error(y_test_inv, test_predict_inv)
    test_r2 = r2_score(y_test_inv, test_predict_inv)

    print(f"\nTest RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}, R-squared: {test_r2:.2f}")

    metrics_to_save["Test RMSE"] = test_rmse
    metrics_to_save["Test MAE"] = test_mae
    metrics_to_save["Test R-squared"] = test_r2

    plt.figure(figsize=(15, 7))
    plt.plot(y_test_inv, label='Actual Glucose (Test)')
    plt.plot(test_predict_inv, label='Predicted Glucose (Test)')
    plt.title('Glucose Level Prediction (Test Set)')
    plt.xlabel('Time Step')
    plt.ylabel('Glucose Level')
    plt.legend()
    plt.grid(True)
    plt.savefig(prediction_plot_path)
    plt.close()
    print(f"Test prediction plot saved to {prediction_plot_path}")

    plot_paths_to_save["Test Predictions Plot"] = prediction_plot_path
    save_statistics_to_file(metrics_to_save, stats_save_path, plot_paths_to_save)

    print("\nTraining and evaluation complete.")
    return model, scaler

if __name__ == "__main__":
    LOOK_BACK = 10
    EPOCHS = 100
    BATCH_SIZE = 32

    # Configure prediction horizon
    PREDICTION_HORIZON_MINUTES = 120
    TIME_INTERVAL_MINUTES = 5 # Your data interval
    PREDICTION_STEPS_AHEAD = PREDICTION_HORIZON_MINUTES // TIME_INTERVAL_MINUTES

    print(f"Model configured to predict {PREDICTION_HORIZON_MINUTES} minutes ({PREDICTION_STEPS_AHEAD} steps) into the future.")

    patientNumber = 559

    base_data_path = os.path.join('..', '..', 'CleanData', 'OhioT1DM')
    training_xml_path = os.path.join(base_data_path, f'{patientNumber}-ws-training.xml')
    testing_xml_path = os.path.join(base_data_path, f'{patientNumber}-ws-testing.xml')

    output_folder_name = f'./120_Min_Predictions/patient-{patientNumber}'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder_path = os.path.join(script_dir, output_folder_name)
    os.makedirs(output_folder_path, exist_ok=True)

    model_save_path = os.path.join(output_folder_path, f'glucose_lstm_model_{patientNumber}.h5')
    stats_save_path = os.path.join(output_folder_path, f'glucose_prediction_statistics_{patientNumber}.txt')
    loss_plot_path = os.path.join(output_folder_path, f'glucose_training_loss_{patientNumber}.png')
    prediction_plot_path = os.path.join(output_folder_path, f'glucose_test_predictions_{patientNumber}.png')

    trained_model, data_scaler = train_glucose_lstm(
        train_xml_file_path=training_xml_path,
        test_xml_file_path=testing_xml_path,
        look_back=LOOK_BACK,
        prediction_steps_ahead=PREDICTION_STEPS_AHEAD,
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
            print(f"Model loaded from {model_save_path}")

            df_test_for_prediction = parse_glucose_data(testing_xml_path)
            # Ensure enough data for the look_back sequence AND the prediction horizon
            required_data_points = LOOK_BACK + PREDICTION_STEPS_AHEAD - 1

            if not df_test_for_prediction.empty and len(df_test_for_prediction) >= required_data_points:
                last_sequence_raw = df_test_for_prediction['glucose_value'].values[-LOOK_BACK:].reshape(-1, 1)
                scaled_last_sequence = data_scaler.transform(last_sequence_raw)
                scaled_last_sequence = np.reshape(scaled_last_sequence, (1, LOOK_BACK, 1))

                predicted_scaled_value = loaded_model.predict(scaled_last_sequence)
                predicted_glucose_value = data_scaler.inverse_transform(predicted_scaled_value)
                print(f"\nPredicted glucose level {PREDICTION_HORIZON_MINUTES} minutes into the future (based on last {LOOK_BACK} test values): {predicted_glucose_value[0][0]:.2f}")
            else:
                print(f"Not enough test data points ({len(df_test_for_prediction)}) to make a prediction for look_back ({LOOK_BACK}) and horizon ({PREDICTION_HORIZON_MINUTES} min). Need at least {required_data_points} data points.")

        except Exception as e:
            print(f"Error loading model or making prediction: {e}")
    else:
        print("\nModel training or data parsing failed. Skipping model loading demonstration.")
