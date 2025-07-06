import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Input, LayerNormalization, Dropout, Conv1D, GlobalAveragePooling1D
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

# --- Custom Keras Layers for Mamba (Simplified Approximation) ---

class MambaBlock(tf.keras.layers.Layer):
    """
    A simplified Keras implementation of a Mamba-like block.
    This approximation focuses on capturing the local context via Conv1D
    and a gating mechanism, rather than the full, complex selective scan.
    It aims to provide a Mamba-inspired alternative to Transformer blocks
    for sequence modeling in Keras.

    Based on the conceptual components of Mamba:
    1. Linear projection (input expansion)
    2. Convolutional layer (local context)
    3. Selective linear projection (simplified delta, A, B, C parameters)
    4. Gating mechanism (SiLU and element-wise multiplication)
    5. Residual connection and Layer Normalization
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout_rate=0.1, **kwargs):
        """
        Initializes the MambaBlock.

        Args:
            d_model (int): The dimension of the input and output embeddings.
            d_state (int): The dimension of the state in the SSM. (Simplified, mostly for conceptual alignment)
            d_conv (int): The kernel size for the 1D convolution.
            expand (int): Expansion factor for the hidden dimension.
            dropout_rate (float): Dropout rate.
        """
        super(MambaBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dropout_rate = dropout_rate

        self.d_inner = int(self.expand * self.d_model)

        # 1. Input projection and expansion
        # Projects input to 2 * d_inner for the GLU-like gating
        self.in_proj = Dense(2 * self.d_inner, use_bias=False, name="input_projection")

        # 2. Convolutional layer (depthwise for local context)
        # Kernel size d_conv, ensures local information capture
        self.conv1d = Conv1D(
            filters=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner, # Depthwise convolution
            padding='causal', # Causal padding to prevent looking into the future
            activation="silu",
            use_bias=False,
            name="conv1d"
        )

        # 3. Selective linear projection (simplified)
        # These layers are a simplified representation of generating SSM parameters (delta, A, B, C)
        # For a full Mamba, these would interact with the state. Here, they contribute to the gate.
        self.x_proj = Dense(self.d_state + self.d_state + self.d_inner, use_bias=False, name="x_projection") # delta, B, C (simplified)

        # Output projection
        self.out_proj = Dense(self.d_model, use_bias=False, name="output_projection")

        # Normalization and Dropout
        self.norm = LayerNormalization(epsilon=1e-6, name="norm")
        self.dropout = Dropout(self.dropout_rate, name="dropout")


    def call(self, inputs, training=None):
        """
        Performs the forward pass of the Mamba-like Block.

        Args:
            inputs (tf.Tensor): The input tensor to the block (batch_size, sequence_length, d_model).
            training (bool, optional): Whether the model is in training mode (affects dropout).
                                       Defaults to None, Keras handles this during fit/predict.
        """
        # Residual connection
        residual = inputs

        # Normalize input
        x = self.norm(inputs)

        # Input projection and split for GLU-like gating
        # x_in_proj will be split into two parts for gating
        x_in_proj = self.in_proj(x)
        x_gate, x_conv_input = tf.split(x_in_proj, num_or_size_splits=2, axis=-1)

        # Convolutional part for local context
        x_conv = self.conv1d(x_conv_input)
        x_conv = tf.nn.silu(x_conv) # SiLU activation after convolution

        # Simplified selective projection (mimicking delta, B, C generation)
        # In a true Mamba, this would involve more complex state manipulation
        # Here, it contributes to the gating mechanism
        x_s_proj = self.x_proj(x_conv)
        # We'll just use the last part for a simplified gate
        # For a full Mamba, delta would be used to update A, B, C
        # Here, we just take a part of the projection for the gate
        _, _, x_selective_gate = tf.split(x_s_proj, num_or_size_splits=[self.d_state, self.d_state, self.d_inner], axis=-1)


        # Gating mechanism (GLU-like)
        # Element-wise multiplication of the input gate with the selective gate
        gated_output = x_gate * tf.nn.silu(x_selective_gate)

        # Output projection
        output = self.out_proj(gated_output)

        # Add residual connection and apply dropout
        output = self.dropout(output, training=training)
        return residual + output

    def get_config(self):
        """
        Returns the serializable configuration of the layer.
        Required for saving and loading models with custom layers.
        """
        config = super(MambaBlock, self).get_config()
        config.update({
            'd_model': self.d_model,
            'd_state': self.d_state,
            'd_conv': self.d_conv,
            'expand': self.expand,
            'dropout_rate': self.dropout_rate,
        })
        return config


# --- Data Processing Functions (Unchanged from original script) ---

def parse_glucose_data(xml_file_path):
    """
    Parses an XML file containing glucose level data and returns a pandas DataFrame.

    Args:
        xml_file_path (str): The path to the XML file.

    Returns:
        pd.DataFrame: A DataFrame with 'timestamp' and 'glucose_value' columns.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_file_path = os.path.abspath(os.path.join(script_dir, xml_file_path))
    print(f"Attempting to open file at absolute path: {absolute_file_path}")

    try:
        tree = ET.parse(absolute_file_path)
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
        print(f"Error: The file '{absolute_file_path}' was not found. Please check the path.")
        print(f"Current working directory (from os.getcwd()): {os.getcwd()}")
        return pd.DataFrame()
    except ET.ParseError as e:
        print(f"Error parsing XML file '{absolute_file_path}': {e}")
        return pd.DataFrame()
    except ValueError as e:
        print(f"Error converting timestamp in '{absolute_file_path}': {e}")
        print("Please check the date format in your XML file. Expected format: DD-MM-YYYY HH:MM:SS")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred while parsing '{absolute_file_path}': {e}")
        return pd.DataFrame()

def create_sequences(data, look_back=1):
    """
    Creates sequences for time series training.
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

# --- Main Script Logic (Modified for Mamba) ---

def train_glucose_mamba(train_xml_file_path, test_xml_file_path, look_back=10, epochs=50, batch_size=32,
                        model_save_path='glucose_mamba_model.h5',
                        stats_save_path='glucose_prediction_statistics_mamba.txt',
                        loss_plot_path='training_loss_mamba.png',
                        prediction_plot_path='test_predictions_mamba.png'):
    """
    Trains a Mamba-like model for glucose level prediction and evaluates it on a separate test set.
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
        return None, None

    print("Training data parsed successfully.")
    print(f"Total training data points: {len(df_train)}")

    # Normalize the training glucose values
    scaler = MinMaxScaler(feature_range=(0, 1))
    glucose_values_train = df_train['glucose_value'].values.reshape(-1, 1)
    scaled_glucose_values_train = scaler.fit_transform(glucose_values_train)

    # Create sequences for training
    X_train, y_train = create_sequences(scaled_glucose_values_train, look_back)
    # Reshape for Mamba: (samples, timesteps, features)
    # Here, features is 1 (glucose value), which will be the d_model for the Mamba block
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    print(f"Training data input shape: {X_train.shape}, target shape: {y_train.shape}")

    # Build the Mamba-like model
    print("\n--- Building Mamba-like Model ---")
    d_model = 1  # Input feature dimension (glucose value)
    d_state = 16 # Dimension of the state in the SSM (conceptual)
    d_conv = 4   # Kernel size for the 1D convolution
    expand_factor = 2 # Expansion factor for the hidden dimension
    dropout_rate = 0.1 # Dropout rate for Mamba layers

    inputs = Input(shape=(look_back, d_model)) # Input shape is (timesteps, features)

    # Add one or more Mamba Blocks
    # Can add multiple blocks by repeating this line: x = MambaBlock(...)(x)
    mamba_block = MambaBlock(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand_factor,
        dropout_rate=dropout_rate
    )
    x = mamba_block(inputs) # The `training` argument is handled implicitly by Keras during fit/predict

    # Global Average Pooling to reduce sequence dimension before the final Dense layer
    # This converts (batch_size, look_back, d_model) to (batch_size, d_model)
    x = GlobalAveragePooling1D()(x)

    # Output layer for regression (predicting a single glucose value)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
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
        # When saving custom layers, Keras needs to know their definitions
        model.save(model_save_path, save_format='h5') # Use h5 format for custom objects
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

    print(f"Training RMSE: {train_rmse:.2f}")
    print(f"Training MAE: {train_mae:.2f}")
    print(f"Training R-squared: {train_r2:.2f}")

    metrics_to_save = {
        "Training RMSE": train_rmse,
        "Training MAE": train_mae,
        "Training R-squared": train_r2
    }

    plot_paths_to_save = {
        "Training Loss Plot": loss_plot_path
    }

    print(f"\n--- Testing Data Processing & Evaluation ---")
    print(f"Parsing testing data from {test_xml_file_path}...")
    df_test = parse_glucose_data(test_xml_file_path)

    if df_test.empty:
        print("Testing data parsing failed or no data found. Cannot evaluate on test set.")
        save_statistics_to_file(metrics_to_save, stats_save_path, plot_paths_to_save)
        return model, scaler

    print("Testing data parsed successfully.")
    print(f"Total testing data points: {len(df_test)}")

    glucose_values_test = df_test['glucose_value'].values.reshape(-1, 1)
    scaled_glucose_values_test = scaler.transform(glucose_values_test)

    X_test, y_test = create_sequences(scaled_glucose_values_test, look_back)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    print(f"Testing data input shape: {X_test.shape}, target shape: {y_test.shape}")

    test_predict = model.predict(X_test)
    test_predict_inv = scaler.inverse_transform(test_predict)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_predict_inv))
    test_mae = mean_absolute_error(y_test_inv, test_predict_inv)
    test_r2 = r2_score(y_test_inv, test_predict_inv)

    print(f"\nTest RMSE: {test_rmse:.2f}")
    print(f"Test MAE: {test_mae:.2f}")
    print(f"Test R-squared: {test_r2:.2f}")

    metrics_to_save["Test RMSE"] = test_rmse
    metrics_to_save["Test MAE"] = test_mae
    metrics_to_save["Test R-squared"] = test_r2

    plt.figure(figsize=(15, 7))
    plt.plot(y_test_inv, label='Actual Glucose Level (Test Set)')
    plt.plot(test_predict_inv, label='Predicted Glucose Level (Test Set)')
    plt.title('Glucose Level Prediction (Test Set) - Mamba-like Model')
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

# Example usage:
if __name__ == "__main__":
    LOOK_BACK = 10
    EPOCHS = 100
    BATCH_SIZE = 32

    patientNum = 559

    base_data_path = os.path.join('..', '..', 'CleanData', 'OhioT1DM')
    training_xml_path = os.path.join(base_data_path, f'{patientNum}-ws-training.xml')
    testing_xml_path = os.path.join(base_data_path, f'{patientNum}-ws-testing.xml')

    output_folder_name = f'patient-{patientNum}-mamba' # Changed folder name
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder_path = os.path.join(script_dir, output_folder_name)

    os.makedirs(output_folder_path, exist_ok=True)
    print(f"Ensured output folder exists at: {output_folder_path}")

    model_save_filename = f'glucose_mamba_model_{patientNum}.h5' # Changed filename
    stats_save_filename = f'glucose_prediction_statistics_{patientNum}_mamba.txt' # Changed filename
    loss_plot_filename = f'glucose_training_loss_{patientNum}_mamba.png' # Changed filename
    prediction_plot_filename = f'glucose_test_predictions_{patientNum}_mamba.png' # Changed filename

    model_save_path = os.path.join(output_folder_path, model_save_filename)
    stats_save_path = os.path.join(output_folder_path, stats_save_filename)
    loss_plot_path = os.path.join(output_folder_path, loss_plot_filename)
    prediction_plot_path = os.path.join(output_folder_path, prediction_plot_filename)

    trained_model, data_scaler = train_glucose_mamba(
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
            # When loading models with custom layers, provide them in custom_objects
            loaded_model = load_model(model_save_path, custom_objects={
                'MambaBlock': MambaBlock
            })
            print(f"Model loaded successfully from {model_save_path}")

            df_test_for_prediction = parse_glucose_data(testing_xml_path)
            if not df_test_for_prediction.empty and len(df_test_for_prediction) >= LOOK_BACK:
                last_sequence_raw = df_test_for_prediction['glucose_value'].values[-LOOK_BACK:].reshape(-1, 1)
                scaled_last_sequence = data_scaler.transform(last_sequence_raw)
                scaled_last_sequence = np.reshape(scaled_last_sequence, (1, LOOK_BACK, 1))

                predicted_scaled_value = loaded_model.predict(scaled_last_sequence)
                predicted_glucose_value = data_scaler.inverse_transform(predicted_scaled_value)
                print(f"\nPredicted next glucose level using loaded Mamba-like model (based on last {LOOK_BACK} test values): {predicted_glucose_value[0][0]:.2f}")
            else:
                print(f"Not enough data points ({len(df_test_for_prediction)}) in the test DataFrame to make a prediction using the defined look_back ({LOOK_BACK}).")

        except Exception as e:
            print(f"Error loading model or making prediction: {e}")
    else:
        print("\nModel training or data parsing failed, skipping model loading demonstration.")
