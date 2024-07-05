import matplotlib.pyplot as plt
import seaborn as sns
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


# dataset = pd.read_csv('hit_data_smaller.csv')
dataset = pd.read_csv('hit_data.csv')
x, y = prepare_data_for_ml(dataset)

#split into 70% training and 30% testing
indices = np.arange(len(x))
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(x, y, indices, test_size=0.3)

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

history = model.fit(X_train_scaled, y_train_scaled, epochs=100, batch_size=32, validation_split=0.2)

#predict params
predicted_params_scaled = model.predict(X_test_scaled)
#puts it back into its original scale for better comparison and understanding 
predicted_params = scaler_y.inverse_transform(predicted_params_scaled)

#this gets the helix ids of the test dataset 
test_helix_ids = dataset['helix_id'].unique()[test_indices]

#gets the specific row of the helix that was used in the test dataset 
rows_in_dataset = [x * 10 for x in test_helix_ids]
test_helix_data = dataset.iloc[rows_in_dataset]

# MSE comparison 
# true_params = test_helix_data[['true_alpha', 'true_kappa', 'true_tan_lambda']].to_numpy() 
# curve_fit_params = test_helix_data[['curve_fit_alpha', 'curve_fit_kappa', 'curve_fit_tan_lambda']].to_numpy()
# ml_params = predicted_params 

true_alpha = test_helix_data['true_alpha'].to_numpy() 
true_kappa = test_helix_data['true_kappa'].to_numpy() 
true_tan_lambda = test_helix_data['true_tan_lambda'].to_numpy() 

curve_fit_alpha = test_helix_data['curve_fit_alpha'].to_numpy() 
curve_fit_kappa = test_helix_data['curve_fit_kappa'].to_numpy() 
curve_fit_tan_lambda = test_helix_data['curve_fit_tan_lambda'].to_numpy() 

ml_alpha = predicted_params[:, 0]
ml_kappa = predicted_params[:, 1]
ml_tan_lambda = predicted_params[:, 2]

#calculate residuals for curve
residual_curve_fit_alpha = true_alpha - curve_fit_alpha
residual_curve_fit_kappa = true_kappa - curve_fit_kappa
residual_curve_fit_tan_lambda = true_tan_lambda - curve_fit_tan_lambda

#calculate residuals for ml predicted
residual_ml_alpha = true_alpha - ml_alpha
residual_ml_kappa = true_kappa - ml_kappa
residual_ml_tan_lambda = true_tan_lambda - ml_tan_lambda

mse_curve_fit_alpha = mean_squared_error(true_alpha, curve_fit_alpha)
mse_curve_fit_kappa = mean_squared_error(true_kappa, curve_fit_kappa)
mse_curve_fit_tan_lambda = mean_squared_error(true_tan_lambda, curve_fit_tan_lambda)

mse_ml_alpha = mean_squared_error(true_alpha, ml_alpha)
mse_ml_kappa = mean_squared_error(true_kappa, ml_kappa)
mse_ml_tan_lambda = mean_squared_error(true_tan_lambda, ml_tan_lambda)

mse_content = f"""
MSE for Curve-Fit Alpha: {mse_curve_fit_alpha}
MSE for Curve-Fit Kappa: {mse_curve_fit_kappa}
MSE for Curve-Fit Tan Lambda: {mse_curve_fit_tan_lambda}

MSE for ML Alpha: {mse_ml_alpha}
MSE for ML Kappa: {mse_ml_kappa}
MSE for ML Tan Lambda: {mse_ml_tan_lambda}
"""

with open("mse_predictions.txt", "w") as file:
    file.write(mse_content)

# file = open("mse_predictons.txt", 'w')
# file.write(f"curve fit MSE: {mse_curve_fit}\n")
# file.write(f"ml MSE: {mse_ml_model}")
# file.close()


# print(f"curve fit MSE: {mse_curve_fit}")
# print(f"ml MSE: {mse_ml_model}")






fig, axes = plt.subplots(3, 2, figsize=(14, 12))

sns.histplot(residual_curve_fit_alpha, kde=True, ax=axes[0, 0], color='green', label='Curve-Fit Residuals', alpha=0.6)
sns.histplot(residual_ml_alpha, kde=True, ax=axes[0, 1], color='red', label='ML Residuals', alpha=0.6)
axes[0, 0].legend()
axes[0, 1].legend()
axes[0, 0].set_title('Alpha: True - Curve-Fit')
axes[0, 1].set_title('alpha: True - ML')

sns.histplot(residual_curve_fit_kappa, kde=True, ax=axes[1, 0], color='green', label='Curve-Fit Residuals', alpha=0.6)
sns.histplot(residual_ml_kappa, kde=True, ax=axes[1, 1], color='red', label='ML Residuals', alpha=0.6)
axes[1, 0].legend()
axes[1, 1].legend()
axes[1, 0].set_title('kappa: True - Curve-Fit')
axes[1, 1].set_title('kappa: True - ML')

sns.histplot(residual_curve_fit_tan_lambda, kde=True, ax=axes[2, 0], color='green', label='Curve-Fit Residuals', alpha=0.6)
sns.histplot(residual_ml_tan_lambda, kde=True, ax=axes[2, 1], color='red', label='ML Residuals', alpha=0.6)
axes[2, 0].legend()
axes[2, 1].legend()
axes[2, 0].set_title('tan lambda: True - Curve-Fit')
axes[2, 1].set_title('tan lambda: True - ML')

plt.tight_layout()
plt.show()