import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Input, LayerNormalization, Dropout, Conv1D, GlobalAveragePooling1D, Concatenate
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

# --- Custom Keras Layers for Mamba (Simplified Approximation) ---

class MambaBlock(tf.keras.layers.Layer):
    """
    A simplified Keras implementation of a Mamba-like block for sequence modeling.
    This block is designed to be a core component of a sequence-to-sequence architecture,
    capturing long-range dependencies in the data.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout_rate=0.1, **kwargs):
        super(MambaBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dropout_rate = dropout_rate

        self.d_inner = int(self.expand * self.d_model)

        # Projections and Convolutional Layers
        self.in_proj = Dense(2 * self.d_inner, use_bias=False, name="input_projection")
        self.conv1d = Conv1D(
            filters=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding='causal',
            activation="silu",
            use_bias=False,
            name="conv1d"
        )
        self.x_proj = Dense(self.d_state + self.d_state + self.d_inner, use_bias=False, name="x_projection")
        self.out_proj = Dense(self.d_model, use_bias=False, name="output_projection")

        # Normalization and Dropout
        self.norm = LayerNormalization(epsilon=1e-6, name="norm")
        self.dropout = Dropout(self.dropout_rate, name="dropout")

    def call(self, inputs, training=None):
        """Forward pass of the Mamba block."""
        residual = inputs
        x = self.norm(inputs)

        # Input projection and split for gating
        x_in_proj = self.in_proj(x)
        x_gate, x_conv_input = tf.split(x_in_proj, num_or_size_splits=2, axis=-1)

        # 1D causal convolution
        x_conv = self.conv1d(x_conv_input)
        x_conv = tf.nn.silu(x_conv)

        # Projection for selective state space model (SSM) part
        x_s_proj = self.x_proj(x_conv)
        _, _, x_selective_gate = tf.split(x_s_proj, num_or_size_splits=[self.d_state, self.d_state, self.d_inner], axis=-1)

        # Gating mechanism
        gated_output = x_gate * tf.nn.silu(x_selective_gate)
        
        # Output projection
        output = self.out_proj(gated_output)

        # Dropout and residual connection
        output = self.dropout(output, training=training)
        return residual + output

    def get_config(self):
        """Serializes the layer's configuration."""
        config = super(MambaBlock, self).get_config()
        config.update({
            'd_model': self.d_model,
            'd_state': self.d_state,
            'd_conv': self.d_conv,
            'expand': self.expand,
            'dropout_rate': self.dropout_rate,
        })
        return config

# --- Data Processing Functions ---

def load_multimodal_data(csv_file_path):
    """Loads and preprocesses multimodal data from a CSV file."""
    if not os.path.exists(csv_file_path):
        print(f"Error: The file '{csv_file_path}' was not found.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_file_path)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values(by='Timestamp').reset_index(drop=True)
        # Ensure we only use numeric data for modeling, and handle potential missing values
        df = df[['Glucose', 'EcgWaveform']].dropna()
        return df
    except Exception as e:
        print(f"An error occurred while loading data from '{csv_file_path}': {e}")
        return pd.DataFrame()


def create_multimodal_sequences(data, look_back=1, prediction_steps_ahead=1):
    """
    Creates sequences for multimodal time series training.
    Predicts a future glucose value based on past multimodal data.
    """
    X, y = [], []
    # The target variable is the 'Glucose' column, which is at index 0
    glucose_data = data[:, 0] 
    
    max_index_needed = look_back + prediction_steps_ahead - 1

    if len(data) <= max_index_needed:
        print(f"Warning: Not enough data points ({len(data)}) for look_back ({look_back}) and prediction horizon ({prediction_steps_ahead} steps).")
        return np.array([]), np.array([])

    for i in range(len(data) - max_index_needed):
        # Input sequence contains all features (Glucose, ECG)
        X.append(data[i:(i + look_back), :])
        # Output is the future glucose value
        y.append(glucose_data[i + look_back + prediction_steps_ahead - 1])
    return np.array(X), np.array(y)


def save_statistics_to_file(metrics_dict, file_path, plot_paths=None):
    """Saves evaluation metrics and plot paths to a text file."""
    try:
        with open(file_path, 'w') as f:
            f.write("--- Multimodal Model Performance Statistics ---\n\n")
            for key, value in metrics_dict.items():
                f.write(f"{key}: {value:.4f}\n")

            if plot_paths:
                f.write("\n--- Associated Plots ---\n")
                for plot_name, plot_path in plot_paths.items():
                    f.write(f"{plot_name}: {os.path.abspath(plot_path)}\n")
        print(f"\nStatistics saved to {file_path}")
    except Exception as e:
        print(f"Error saving statistics to file {file_path}: {e}")

# --- Main Script Logic ---

def train_multimodal_mamba(df_train, df_test, look_back=10, prediction_steps_ahead=1, epochs=50, batch_size=32,
                           model_save_path='multimodal_mamba_model.h5',
                           stats_save_path='multimodal_prediction_stats.txt',
                           loss_plot_path='training_loss_multimodal.png',
                           prediction_plot_path='test_predictions_multimodal.png'):
    """Trains and evaluates a multimodal Mamba-like model for glucose prediction."""
    print("--- Training Data Preprocessing ---")
    if df_train.empty:
        print("Training DataFrame is empty. Exiting training.")
        return None, None, None

    # Use separate scalers for each modality
    glucose_scaler = MinMaxScaler(feature_range=(0, 1))
    ecg_scaler = MinMaxScaler(feature_range=(0, 1))

    # Create copies to avoid SettingWithCopyWarning
    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()

    # Fit scalers on training data and transform both train and test sets
    df_train_scaled['Glucose_scaled'] = glucose_scaler.fit_transform(df_train[['Glucose']])
    df_train_scaled['EcgWaveform_scaled'] = ecg_scaler.fit_transform(df_train[['EcgWaveform']])
    
    scaled_data_train = df_train_scaled[['Glucose_scaled', 'EcgWaveform_scaled']].values

    X_train, y_train = create_multimodal_sequences(scaled_data_train, look_back, prediction_steps_ahead)
    if X_train.size == 0 or y_train.size == 0:
        print("Training sequences could not be created. Exiting.")
        return None, None, None

    print(f"Training input shape: {X_train.shape}, target shape: {y_train.shape}")

    print("\n--- Building Multimodal Mamba-like Model ---")
    num_features = X_train.shape[2] # Should be 2 (Glucose, ECG)
    d_model = 64 # Increased model dimension for more capacity
    d_state = 16
    d_conv = 4
    expand_factor = 2
    dropout_rate = 0.1

    inputs = Input(shape=(look_back, num_features))
    
    x = Dense(d_model)(inputs)

    mamba_block = MambaBlock(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand_factor,
        dropout_rate=dropout_rate
    )
    x = mamba_block(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()

    print("\n--- Training Model ---")
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1, shuffle=False)
    print("Model training complete.")

    try:
        model.save(model_save_path, save_format='h5')
        print(f"\nModel saved to {model_save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Multimodal Model Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Training loss plot saved to {loss_plot_path}")

    print(f"\n--- Testing Data Processing & Evaluation ---")
    if df_test.empty:
        print("Testing DataFrame is empty. Skipping test evaluation.")
        return model, glucose_scaler, ecg_scaler

    df_test_scaled['Glucose_scaled'] = glucose_scaler.transform(df_test[['Glucose']])
    df_test_scaled['EcgWaveform_scaled'] = ecg_scaler.transform(df_test[['EcgWaveform']])
    scaled_data_test = df_test_scaled[['Glucose_scaled', 'EcgWaveform_scaled']].values
    
    X_test, y_test = create_multimodal_sequences(scaled_data_test, look_back, prediction_steps_ahead)
    if X_test.size == 0 or y_test.size == 0:
        print("Testing sequences could not be created. Skipping test evaluation.")
        return model, glucose_scaler, ecg_scaler

    print(f"Testing input shape: {X_test.shape}, target shape: {y_test.shape}")

    test_predict_scaled = model.predict(X_test)
    test_predict_inv = glucose_scaler.inverse_transform(test_predict_scaled)
    y_test_inv = glucose_scaler.inverse_transform(y_test.reshape(-1, 1))

    test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_predict_inv))
    test_mae = mean_absolute_error(y_test_inv, test_predict_inv)
    test_r2 = r2_score(y_test_inv, test_predict_inv)

    print(f"\nTest RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}, R-squared: {test_r2:.2f}")
    
    metrics_to_save = {
        "Test RMSE": test_rmse, "Test MAE": test_mae, "Test R-squared": test_r2
    }

    plt.figure(figsize=(15, 7))
    plt.plot(y_test_inv, label='Actual Glucose (Test)')
    plt.plot(test_predict_inv, label='Predicted Glucose (Test)', alpha=0.8)
    plt.title('Multimodal Glucose Prediction (Test Set) - Mamba-like Model')
    plt.xlabel('Time Step')
    plt.ylabel('Glucose Level')
    plt.legend()
    plt.grid(True)
    plt.savefig(prediction_plot_path)
    plt.close()
    print(f"Test prediction plot saved to {prediction_plot_path}")
    
    plot_paths_to_save = {"Test Predictions Plot": prediction_plot_path, "Training Loss Plot": loss_plot_path}
    save_statistics_to_file(metrics_to_save, stats_save_path, plot_paths_to_save)

    print("\nTraining and evaluation complete.")
    return model, glucose_scaler, ecg_scaler

# --- Example Usage ---
if __name__ == "__main__":
    # --- Configuration ---
    LOOK_BACK = 12
    EPOCHS = 75
    BATCH_SIZE = 32
    PREDICTION_HORIZON_MINUTES = 30
    TIME_INTERVAL_MINUTES = 1
    PREDICTION_STEPS_AHEAD = PREDICTION_HORIZON_MINUTES // TIME_INTERVAL_MINUTES

    print(f"Model configured to use {LOOK_BACK*TIME_INTERVAL_MINUTES} minutes of past data to predict {PREDICTION_HORIZON_MINUTES} minutes into the future.")

    # --- File Path and Data Splitting ---
    full_data_path = './Clean_Data/Patient_1_merged_data.csv'
    
    full_df = load_multimodal_data(full_data_path)

    if not full_df.empty:
        # Splitting data into 80% training and 20% testing
        train_size = int(len(full_df) * 0.8)
        df_train = full_df[:train_size]
        df_test = full_df[train_size:]
        print(f"Data loaded and split into {len(df_train)} training samples and {len(df_test)} testing samples.")

        # --- Output Paths ---
        output_folder_name = f'./{PREDICTION_HORIZON_MINUTES}_Min_Multimodal_Predictions_Patient_1'
        os.makedirs(output_folder_name, exist_ok=True)
        print(f"Output folder created at: {output_folder_name}")

        model_save_path = os.path.join(output_folder_name, f'multimodal_mamba_model_{PREDICTION_HORIZON_MINUTES}min.h5')
        stats_save_path = os.path.join(output_folder_name, f'multimodal_stats_{PREDICTION_HORIZON_MINUTES}min.txt')
        loss_plot_path = os.path.join(output_folder_name, f'multimodal_loss_{PREDICTION_HORIZON_MINUTES}min.png')
        prediction_plot_path = os.path.join(output_folder_name, f'multimodal_predictions_{PREDICTION_HORIZON_MINUTES}min.png')

        # --- Run Training and Evaluation ---
        trained_model, g_scaler, e_scaler = train_multimodal_mamba(
            df_train=df_train,
            df_test=df_test,
            look_back=LOOK_BACK,
            prediction_steps_ahead=PREDICTION_STEPS_AHEAD,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            model_save_path=model_save_path,
            stats_save_path=stats_save_path,
            loss_plot_path=loss_plot_path,
            prediction_plot_path=prediction_plot_path
        )

        # --- Demonstrate Loading and Prediction ---
        if trained_model:
            print("\n--- Demonstrating Model Loading and Prediction ---")
            try:
                loaded_model = load_model(model_save_path, custom_objects={'MambaBlock': MambaBlock})
                print(f"Model loaded successfully from {model_save_path}")

                if not df_test.empty and len(df_test) >= LOOK_BACK:
                    last_sequence_raw = df_test.tail(LOOK_BACK)
                    
                    last_sequence_scaled = pd.DataFrame()
                    last_sequence_scaled['Glucose'] = g_scaler.transform(last_sequence_raw[['Glucose']])[:,0]
                    last_sequence_scaled['EcgWaveform'] = e_scaler.transform(last_sequence_raw[['EcgWaveform']])[:,0]
                    
                    input_sequence = np.reshape(last_sequence_scaled.values, (1, LOOK_BACK, 2))

                    predicted_scaled_value = loaded_model.predict(input_sequence)
                    predicted_glucose_value = g_scaler.inverse_transform(predicted_scaled_value)
                    print(f"\nPredicted glucose level {PREDICTION_HORIZON_MINUTES} minutes into the future (based on last {LOOK_BACK} test values): {predicted_glucose_value[0][0]:.2f}")
                else:
                    print(f"Not enough test data points to make a prediction.")

            except Exception as e:
                print(f"Error loading model or making prediction: {e}")
        else:
            print("\nModel training failed. Skipping model loading demonstration.")
    else:
        print(f"Failed to load data from {full_data_path}. Please check the file path and format.")

