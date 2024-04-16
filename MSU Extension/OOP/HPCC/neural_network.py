import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from generate_simulation_data import *
from ml_functions import *

# Simulation parameters
num_samples = 100
grid_size = 100
num_snapshots = 50
num_steps = 1000
L = 10
dx = L / grid_size
dt = 0.001

# Generate data
training_data, IC_types_train, IC_params_train, IC_equilibria_train, IC_pertubration_amplitudes_train = run_simulation_with_dynamic_conditions(num_samples, grid_size, num_snapshots, num_steps, L, dx, dt)
testing_data, IC_types_test, IC_params_test, IC_equilibria_test, IC_pertubration_amplitudes_test = run_simulation_with_dynamic_conditions(num_samples, grid_size, num_snapshots, num_steps, L, dx, dt)

####################
### Create Model ###
####################

# Define dimensions based on your data
num_samples = training_data.shape[0]
num_channels = training_data.shape[1]
num_snapshots = training_data.shape[2]
num_points = training_data.shape[3]

# Normalize the data
mean = np.mean(training_data, axis=(0, 2, 3), keepdims=True)
std = np.std(training_data, axis=(0, 2, 3), keepdims=True)

training_data_normalized = (training_data - mean) / std
testing_data_normalized = (testing_data - mean) / std

# Model architecture
model = keras.Sequential([
    # Input layer
    layers.Input(shape=(num_channels, num_snapshots, num_points)),
    # Reshape to data to optimal shape for 1D CNN
    layers.Reshape((num_snapshots, num_points, num_channels)),
    # 1D CNN for Spatial Feature Extraction
    layers.TimeDistributed(layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')),
    # Reshape to combine channels after 1D Conv
    layers.Reshape((num_snapshots, num_points * 32)),
    # Reshape for applying 2D convolutions, treating each time snapshot as an individual "image"
    layers.Reshape((num_snapshots, num_points, 32, 1)),
    # 2D CNN for Combined Spatial Features
    layers.TimeDistributed(layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')),
    # Flatten to feed into LSTM
    layers.Reshape((num_snapshots, -1)),
    # LSTM for Temporal Analysis
    layers.LSTM(128),
    # Dense Layer for Output
    layers.Dense(num_channels * num_snapshots * num_points),
    # Reshape to output shape
    layers.Reshape((num_channels, num_snapshots, num_points))
])

# Print model summary for debugging
model.summary()

#######################################
### Specify Model Callbacks and Run ###
#######################################

# Define the ReduceLROnPlateau callback
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,  # Reduction factor; new_lr = lr * factor
    patience=5,  # Number of epochs with no improvement after which learning rate will be reduced
    min_lr=1e-6,  # Lower bound on the learning rate
    verbose=1  # If set to 1, the method will print messages when reducing the learning rate
)

# Early Stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True)

# ModelCheckpoint to save the best model during training
model_checkpoint = ModelCheckpoint(
    'model.h5',
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

# CSVLogger to log training data
csv_logger = CSVLogger('training_log.csv', append=False)

# Compile the model
model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

# Fit the model with the learning rate scheduler
history = model.fit(
    training_data_normalized, 
    training_data_normalized,  # Replace with actual targets if different from the input
    epochs=100,
    batch_size=32, 
    validation_split=0.2,
    callbacks=[reduce_lr, early_stopping, model_checkpoint, csv_logger]
)

# Evaluate model
test_metrics = model.evaluate(testing_data_normalized, testing_data_normalized, return_dict=True)
predictions = model.predict(testing_data_normalized)

###########################
### Save Run Information ##
###########################

simulation_data_path = 'simulation_data.npz'
model_path = 'model.h5'
history_path = 'history.pkl'
predictions_path = 'predictions.pkl'
metrics_path = 'metrics.pkl'

# Save Data
np.savez(simulation_data_path, training_data=training_data, testing_data=testing_data)

# Save Model
model.save(model_path)

# Save History
with open(history_path, 'wb') as f:
    pickle.dump(history.history, f)

# Save Predictions
with open(predictions_path, 'wb') as f:
    pickle.dump(predictions, f)

# Save Metrics
with open(metrics_path, 'wb') as f:
    pickle.dump(test_metrics, f)