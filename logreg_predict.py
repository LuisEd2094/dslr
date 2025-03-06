import numpy as np
import pandas as pd

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def main():
    # Load test data and preprocessing parameters
    test_df = pd.read_csv('dataset_test.csv')
    params = np.load('weights.npz', allow_pickle=True)
    
    # Reconstruct feature_means as a pandas Series
    feature_means = pd.Series(
        params['feature_means'], 
        index=params['feature_means_index'].tolist()
    )
    feature_stds = params['feature_stds']
    weights_dict = params['weights'].item()
    label_encoder = params['label_encoder'].tolist()
    
    # Preprocess test data
    test_df = test_df.drop(['Index', 'First Name', 'Last Name', 'Birthday', 'Best Hand'], axis=1)
    
    # Align columns with training data
    required_columns = feature_means.index.tolist()  # No "Best Hand" anymore
    test_df = test_df[required_columns]
    
    # Fill missing values and standardize
    test_df[required_columns] = test_df[required_columns].fillna(feature_means)
    test_df[required_columns] = (test_df[required_columns] - feature_means) / feature_stds
    
    # Predict
    X_test = test_df.values
    houses = label_encoder
    probabilities = np.zeros((X_test.shape[0], len(houses)))
    for i, house in enumerate(houses):
        z = X_test.dot(weights_dict[house])
        probabilities[:, i] = sigmoid(z)
    predicted_houses = [houses[np.argmax(row)] for row in probabilities]
    
    # Save predictions
    result_df = pd.DataFrame({'Index': test_df.index, 'Hogwarts House': predicted_houses})
    result_df.to_csv('houses.csv', index=False)

if __name__ == '__main__':
    main()