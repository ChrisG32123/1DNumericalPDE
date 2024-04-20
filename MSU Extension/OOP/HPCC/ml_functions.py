import pickle
import numpy as np
from tensorflow.keras.models import load_model

# Function to save all relevant model data
def save_model_data(model, history, predictions, test_metrics, model_path='final_model.h5', history_path='training_history.pkl', results_path='model_results.pkl'):
    # Save the model
    model.save(model_path)
    
    # Save the training history
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    
    # Save the test loss, predictions, and other metrics
    results = {
        'test_metrics': test_metrics,
        'predictions': predictions
    }
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
        
# Function to load all relevant model data
def load_model_data(model_path='final_model.h5', history_path='training_history.pkl', results_path='model_results.pkl'):
    # Load the model
    model = load_model(model_path)
    
    # Load the training history
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    
    # Load the results which include test loss, predictions, and metrics
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    return model, history, results

# Function to create training and testing datasets
def create_dataset(dataset, target_dataset, past_steps, num_predicted_snapshots, is_train=True):
    num_instances, num_snapshots, num_fields, num_points = dataset.shape
    num_features = num_fields * num_points
    dataX, dataY = [], []

    if is_train:
        for instance in range(num_instances):
            for start in range(num_snapshots - past_steps - num_predicted_snapshots + 1):
                end = start + past_steps
                seq_x = dataset[instance, start:end, :, :].reshape(past_steps, num_fields, num_points, 1)
                seq_y = dataset[instance, end:end + num_predicted_snapshots, :, :].reshape(num_predicted_snapshots, num_features)
                dataX.append(seq_x)
                dataY.append(seq_y)
    else:
        for instance in range(num_instances):
            # Ensure there are enough snapshots to form a sequence for test data
            if num_snapshots >= past_steps:
                seq_x = dataset[instance, -past_steps:, :, :].reshape(1, past_steps, num_fields, num_points, 1)
                seq_y = target_dataset[instance, 0, :, :].flatten().reshape(1, num_features)
                dataX.append(seq_x)
                dataY.append(seq_y)

    # Separate output fields for separate physics constraints
    return np.array(dataX).reshape(-1, past_steps, num_fields, num_points, 1), np.array(dataY).reshape(-1, num_points, num_fields)


# One Step Ahead
def forecast_snapshots(model, initial_data, future_snapshots, num_instances, num_fields, num_snapshots, num_points):
    output_data = np.zeros((num_instances, num_fields, num_snapshots + future_snapshots - 1, num_points))
    output_data[:, :, :num_snapshots - 1, :] = initial_data

    for i in range(future_snapshots):
        new_input = output_data[:, :, -num_snapshots + 1:, :]  # Select the most recent snapshots for prediction
        new_input_reshaped = new_input.reshape(num_instances, num_fields, num_snapshots - 1, num_points)
        next_snapshot = model.predict(new_input_reshaped)
        next_snapshot_reshaped = next_snapshot.reshape(num_instances, num_fields, num_points)
        output_data[:, :, num_snapshots - 1 + i, :] = next_snapshot_reshaped
    
    return output_data

