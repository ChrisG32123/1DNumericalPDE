import pickle
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