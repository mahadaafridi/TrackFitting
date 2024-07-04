import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from helixfitter import HelixFitter
from sklearn.metrics import mean_squared_error
import ast

def prepare_data_for_ml(dataset):
    """Takes in a pandas dataframe. It groups up the hits for each helix in x, and
    groups up the true parameters (alpha, kappa, tan_lambda) in y. 

    Args:
        dataset (pd.Dataframe): Contains the data from generating hits.

    Returns:
        _type: tuple(numpy_array, numpy_array): split up X and y data for ml model
    """
    X = []
    y = []
    for helix_id in dataset['helix_id'].unique():
        helix_data = dataset[dataset['helix_id'] == helix_id]
        hits = helix_data[['x', 'y', 'z']].values.flatten()  
        #contains all the hits for the helix 
        X.append(hits)
        #contains the true parameter values
        y.append(helix_data[['true_alpha', 'true_kappa', 'true_tan_lambda']].iloc[0].values)
        
    return np.array(X), np.array(y)

def create_ff_model(input_shape, output_shape):
    """Creates a forward feeding neural network. 
    Its shape is the input shape(the hits) -> 64 -> 32 -> 16 -> output_shape(3)

    Args:
        input_shape (int): shape of the input, which is the hits 
        output_shape (int: shape of the output, which is the true parameters 

    Returns:
        _type_: tf.keras.Sequential
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(output_shape)
    ])


dataset = pd.read_csv('hit_data.csv')
x, y = prepare_data_for_ml(dataset)

#split into 70% training and 30% testing
indices = np.arange(len(x))
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    x, y, indices, test_size=0.3)

#scale the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

#create ff model
input_shape = X_train.shape[1]
output_shape = y_train.shape[1]
model = create_ff_model(input_shape, output_shape)
model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train_scaled, y_train_scaled, 
                    epochs=100, 
                    batch_size=32, 
                    validation_split=0.2,
                    verbose=0)

#predict params
predicted_params_scaled = model.predict(X_test_scaled)
predicted_params = scaler_y.inverse_transform(predicted_params_scaled)

#this gets the helix ids of the test dataset 
test_helix_ids = dataset['helix_id'].unique()[test_indices]

#gets the specific row of the helix that was used in the test dataset 
rows_in_dataset = [x * 10 for x in test_helix_ids]
test_helix_data = dataset.iloc[rows_in_dataset]

# MSE comparison 
true_params = test_helix_data[['true_alpha', 'true_kappa', 'true_tan_lambda']].to_numpy()
curve_fit_params = test_helix_data[['curve_fit_alpha', 'curve_fit_kappa', 'curve_fit_tan_lambda']].to_numpy()
ml_params = predicted_params

# print("true")
# print(true_params)
# print("curve")
# print(curve_fit_params)
# print("ml")
# print(ml_params)

mse_curve_fit = mean_squared_error(true_params, curve_fit_params)
mse_ml_model = mean_squared_error(true_params, ml_params)

print(f"curve fit MSE: {mse_curve_fit}")
print(f"ml MSE: {mse_ml_model}")
