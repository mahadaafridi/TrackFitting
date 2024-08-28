import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from helixfitter import HelixFitter

LOGS_DIR = "logs"


# ------------------------------ logging setup ------------------------------
# sets up the logging for the runs
def setup_logging():
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        filename=os.path.join(LOGS_DIR, f"logfile_{datetime.now()}.txt"),
        encoding="utf-8",
        filemode="w",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    return logger


logger = setup_logging()


# ------------------------------ data preperation ------------------------------
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
    for helix_id in dataset["helix_id"].unique():
        helix_data = dataset[dataset["helix_id"] == helix_id]
        hits = helix_data[["x", "y", "z"]].values.flatten()
        # contains all the hits for the helix
        X.append(hits)
        # contains the true parameter values
        y.append(
            helix_data[["true_alpha", "true_kappa", "true_tan_lambda"]].iloc[0].values
        )

    return np.array(X), np.array(y)


def split_and_scale_data(x, y):
    """
    Splits and scales the parameters individually. It then returns the scalers and indicies
    """
    # split into 70% training and 30% testing
    indices = np.arange(len(x))
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        x, y, indices, test_size=0.3
    )

    # scale the data
    scaler_X = StandardScaler()

    # Separate scalers for each target variable y (alpha, kappa, tan_lambda)
    scaler_alpha = StandardScaler()
    scaler_kappa = StandardScaler()
    scaler_tan_lambda = StandardScaler()

    # Scale the input features
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Scale the target variables separately
    y_train_alpha_scaled = scaler_alpha.fit_transform(y_train[:, 0].reshape(-1, 1))
    y_train_kappa_scaled = scaler_kappa.fit_transform(y_train[:, 1].reshape(-1, 1))
    y_train_tan_lambda_scaled = scaler_tan_lambda.fit_transform(
        y_train[:, 2].reshape(-1, 1)
    )

    y_test_alpha_scaled = scaler_alpha.transform(y_test[:, 0].reshape(-1, 1))
    y_test_kappa_scaled = scaler_kappa.transform(y_test[:, 1].reshape(-1, 1))
    y_test_tan_lambda_scaled = scaler_tan_lambda.transform(y_test[:, 2].reshape(-1, 1))

    # Combine the scaled target variables back into a single array
    y_train_scaled = np.hstack(
        [y_train_alpha_scaled, y_train_kappa_scaled, y_train_tan_lambda_scaled]
    )
    y_test_scaled = np.hstack(
        [y_test_alpha_scaled, y_test_kappa_scaled, y_test_tan_lambda_scaled]
    )

    return (
        X_train_scaled,
        y_train_scaled,
        X_test_scaled,
        scaler_alpha,
        scaler_kappa,
        scaler_tan_lambda,
        test_indices,
    )


def get_rows_from_indices(df, indices, range_size=10):
    """
    Gets the rows of the indicies given. The purpose of this is that given an indicie, getting the next 10 rows of that indice means that it will have all the hits for that helix
    """
    result = []
    for index in indices:
        result.append(df.iloc[index : index + range_size])
    return result


# ------------------------------ model def and training ------------------------------
def create_ff_model(input_shape, output_shape):
    """Creates a forward feeding neural network.
    Its shape is the input shape(the hits) -> 64 -> 32 -> 16 -> output_shape(3)

    Args:
        input_shape (int): shape of the input, which is the hits
        output_shape (int: shape of the output, which is the true parameters

    Returns:
        _type_: tf.keras.Sequential
    """
    # lower and upper bounds as the curve fit was also given this
    lower_bounds = [1.5, 0.01, 0.1]
    upper_bounds = [2.0, 0.02, 1.0]

    constraints = []
    for lower, upper in zip(lower_bounds, upper_bounds):
        constraints.append(
            tf.keras.constraints.MinMaxNorm(min_value=lower, max_value=upper)
        )

    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu", input_shape=(input_shape,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(output_shape),
        ]
    )


def train_model(X_train_scaled, y_train_scaled):
    """
    Creates and trains the model. Returns the fitted model
    """
    model = create_ff_model(
        input_shape=X_train_scaled.shape[1], output_shape=y_train_scaled.shape[1]
    )
    model.compile(optimizer="adam", loss="mse")

    history = model.fit(
        X_train_scaled, y_train_scaled, epochs=100, batch_size=32, validation_split=0.2
    )

    return model, history


# ------------------------------ eval ------------------------------
def inverse_transform_predictions(
    X_test_scaled, scaler_alpha, scaler_kappa, scaler_tan_lambda, model
):
    """
    Reverts the scaling of the parameters. The purpose is so that it makes more sense when comparing/observing the data
    """
    predicted_params_scaled = model.predict(X_test_scaled)

    # Inverse transform each of the predicted parameters
    predicted_alpha = scaler_alpha.inverse_transform(
        predicted_params_scaled[:, 0].reshape(-1, 1)
    ).flatten()
    predicted_kappa = scaler_kappa.inverse_transform(
        predicted_params_scaled[:, 1].reshape(-1, 1)
    ).flatten()
    predicted_tan_lambda = scaler_tan_lambda.inverse_transform(
        predicted_params_scaled[:, 2].reshape(-1, 1)
    ).flatten()

    # Combine the inverse transformed predictions back into a single array for analysis
    predicted_params = np.vstack(
        [predicted_alpha, predicted_kappa, predicted_tan_lambda]
    ).T
    return predicted_params


def get_test_data(dataset, test_indices):
    test_helix_ids = dataset["helix_id"].unique()[test_indices]

    # gets the specific row of the helix that was used in the test dataset
    rows_in_dataset = [x * 10 for x in test_helix_ids]
    test_helix_data = dataset.iloc[rows_in_dataset]
    return rows_in_dataset, test_helix_data


def extract_curve_fit_and_ml_data(
    get_rows_from_indices, dataset, predicted_params, rows_in_dataset, test_helix_data
):
    true_alpha = test_helix_data["true_alpha"].to_numpy()
    true_kappa = test_helix_data["true_kappa"].to_numpy()
    true_tan_lambda = test_helix_data["true_tan_lambda"].to_numpy()

    curve_fit_alpha = test_helix_data["curve_fit_alpha"].to_numpy()
    curve_fit_kappa = test_helix_data["curve_fit_kappa"].to_numpy()
    curve_fit_tan_lambda = test_helix_data["curve_fit_tan_lambda"].to_numpy()

    ml_alpha = predicted_params[:, 0]
    ml_kappa = predicted_params[:, 1]
    ml_tan_lambda = predicted_params[:, 2]

    extracted_rows = get_rows_from_indices(dataset, rows_in_dataset)
    curve_fit_chi_sq = test_helix_data["curve_fit_chi_sq"].to_numpy()
    return (
        true_alpha,
        true_kappa,
        true_tan_lambda,
        curve_fit_alpha,
        curve_fit_kappa,
        curve_fit_tan_lambda,
        ml_alpha,
        ml_kappa,
        ml_tan_lambda,
        extracted_rows,
        curve_fit_chi_sq,
    )


def chi_squared_for_ml(predicted_params, extracted_rows):
    ml_chi_squared = []
    for i, param in enumerate(predicted_params):
        alpha, kappa, tan_lambda = param

        hf_ml = HelixFitter()
        # pass in parameters to function to generate hits from it
        ml_track_hits = hf_ml.generate_helix_hits(alpha, kappa, tan_lambda)

        r_predicted = np.sqrt(ml_track_hits[:, 0] ** 2 + ml_track_hits[:, 1] ** 2)
        r_actual = extracted_rows[i]["r"]

        residuals = r_actual - r_predicted
        chi_squared = np.sum(residuals**2)

        ml_chi_squared.append(chi_squared)
    return ml_chi_squared


def calc_residuals(
    true_alpha,
    true_kappa,
    true_tan_lambda,
    curve_fit_alpha,
    curve_fit_kappa,
    curve_fit_tan_lambda,
    ml_alpha,
    ml_kappa,
    ml_tan_lambda,
    curve_fit_chi_sq,
    ml_chi_squared,
):

    # convert to numpy array
    curve_fit_chi_sq = np.array(curve_fit_chi_sq)
    ml_chi_squared = np.array(ml_chi_squared)

    curve_fit_mean = np.mean(curve_fit_chi_sq)
    ml_mean = np.mean(ml_chi_squared)

    curve_fit_std = np.std(curve_fit_chi_sq)
    ml_std = np.std(ml_chi_squared)

    # calculate residuals for curve
    residual_curve_fit_alpha = true_alpha - curve_fit_alpha
    residual_curve_fit_kappa = true_kappa - curve_fit_kappa
    residual_curve_fit_tan_lambda = true_tan_lambda - curve_fit_tan_lambda

    # calculate residuals for ml predicted
    residual_ml_alpha = true_alpha - ml_alpha
    residual_ml_kappa = true_kappa - ml_kappa
    residual_ml_tan_lambda = true_tan_lambda - ml_tan_lambda

    mse_curve_fit_alpha = mean_squared_error(true_alpha, curve_fit_alpha)
    mse_curve_fit_kappa = mean_squared_error(true_kappa, curve_fit_kappa)
    mse_curve_fit_tan_lambda = mean_squared_error(true_tan_lambda, curve_fit_tan_lambda)

    mse_ml_alpha = mean_squared_error(true_alpha, ml_alpha)
    mse_ml_kappa = mean_squared_error(true_kappa, ml_kappa)
    mse_ml_tan_lambda = mean_squared_error(true_tan_lambda, ml_tan_lambda)
    return (
        curve_fit_chi_sq,
        ml_chi_squared,
        curve_fit_mean,
        ml_mean,
        curve_fit_std,
        ml_std,
        residual_curve_fit_alpha,
        residual_curve_fit_kappa,
        residual_curve_fit_tan_lambda,
        residual_ml_alpha,
        residual_ml_kappa,
        residual_ml_tan_lambda,
        mse_curve_fit_alpha,
        mse_curve_fit_kappa,
        mse_curve_fit_tan_lambda,
        mse_ml_alpha,
        mse_ml_kappa,
        mse_ml_tan_lambda,
    )


# ------------------------------ visualization ------------------------------
def write_stats(
    curve_fit_mean,
    ml_mean,
    curve_fit_std,
    ml_std,
    mse_curve_fit_alpha,
    mse_curve_fit_kappa,
    mse_curve_fit_tan_lambda,
    mse_ml_alpha,
    mse_ml_kappa,
    mse_ml_tan_lambda,
    output_file_path,
):
    mse_content = f"""
MSE for Curve-Fit Alpha: {mse_curve_fit_alpha}
MSE for Curve-Fit Kappa: {mse_curve_fit_kappa}
MSE for Curve-Fit Tan Lambda: {mse_curve_fit_tan_lambda}
MSE for ML Alpha: {mse_ml_alpha}
MSE for ML Kappa: {mse_ml_kappa}
MSE for ML Tan Lambda: {mse_ml_tan_lambda}
"""

    # for output file
    chi_squared_stats = f"""
Curve Fit Chi-Squared Values:
Mean: {curve_fit_mean}, Standard Deviation: {curve_fit_std}
ML Model Chi-Squared Values:
Mean: {ml_mean}, Standard Deviation: {ml_std}\n
"""

    with open(output_file_path, "w") as file:
        file.write(chi_squared_stats)
        file.write(mse_content)


def plot_chi_squared(curve_fit_chi_sq, ml_chi_squared):
    """Plots the chi squared of the ml and curve fit models"""
    plt.figure(figsize=(10, 6))

    min_bin = min(curve_fit_chi_sq.min(), ml_chi_squared.min())
    max_bin = max(curve_fit_chi_sq.max(), ml_chi_squared.max())

    bins = np.logspace(np.log10(min_bin), np.log10(max_bin), 20)

    plt.hist(
        curve_fit_chi_sq,
        bins=bins,
        alpha=0.5,
        label="Curve Fit Chi-Squared",
        color="blue",
    )
    plt.hist(
        ml_chi_squared,
        bins=bins,
        alpha=0.5,
        label="ML Model Chi-Squared",
        color="orange",
    )

    plt.xscale("log")
    plt.xlabel("Chi-Squared Values (Log Scale)")
    plt.ylabel("Frequency")
    plt.title("Comparison of Chi-Squared Values")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()


def plot_residuals(
    residual_curve_fit_alpha,
    residual_curve_fit_kappa,
    residual_curve_fit_tan_lambda,
    residual_ml_alpha,
    residual_ml_kappa,
    residual_ml_tan_lambda,
):
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    sns.histplot(
        residual_curve_fit_alpha,
        kde=True,
        ax=axes[0, 0],
        color="green",
        label="Curve-Fit Residuals",
        alpha=0.6,
    )
    sns.histplot(
        residual_ml_alpha,
        kde=True,
        ax=axes[0, 1],
        color="red",
        label="ML Residuals",
        alpha=0.6,
    )
    axes[0, 0].legend()
    axes[0, 1].legend()
    axes[0, 0].set_title("Alpha: True - Curve-Fit")
    axes[0, 1].set_title("alpha: True - ML")

    sns.histplot(
        residual_curve_fit_kappa,
        kde=True,
        ax=axes[1, 0],
        color="green",
        label="Curve-Fit Residuals",
        alpha=0.6,
    )
    sns.histplot(
        residual_ml_kappa,
        kde=True,
        ax=axes[1, 1],
        color="red",
        label="ML Residuals",
        alpha=0.6,
    )
    axes[1, 0].legend()
    axes[1, 1].legend()
    axes[1, 0].set_title("kappa: True - Curve-Fit")
    axes[1, 1].set_title("kappa: True - ML")

    sns.histplot(
        residual_curve_fit_tan_lambda,
        kde=True,
        ax=axes[2, 0],
        color="green",
        label="Curve-Fit Residuals",
        alpha=0.6,
    )
    sns.histplot(
        residual_ml_tan_lambda,
        kde=True,
        ax=axes[2, 1],
        color="red",
        label="ML Residuals",
        alpha=0.6,
    )
    axes[2, 0].legend()
    axes[2, 1].legend()
    axes[2, 0].set_title("tan lambda: True - Curve-Fit")
    axes[2, 1].set_title("tan lambda: True - ML")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data_file = "100,000"
    stats_file = "100,000"

    data_file_path = os.path.join("data", data_file + ".csv")
    stats_file_path = os.path.join("res", data_file + ".txt")
    dataset = pd.read_csv(data_file_path)

    x, y = prepare_data_for_ml(dataset)

    (
        X_train_scaled,
        y_train_scaled,
        X_test_scaled,
        scaler_alpha,
        scaler_kappa,
        scaler_tan_lambda,
        test_indices,
    ) = split_and_scale_data(x, y)

    # Create and train the model
    model, history = train_model(
        X_train_scaled=X_train_scaled, y_train_scaled=y_train_scaled
    )

    # Predict and inverse transform
    predicted_params = inverse_transform_predictions(
        X_test_scaled, scaler_alpha, scaler_kappa, scaler_tan_lambda, model
    )

    # this gets the helix ids of the test dataset
    rows_in_dataset, test_helix_data = get_test_data(dataset, test_indices)

    (
        true_alpha,
        true_kappa,
        true_tan_lambda,
        curve_fit_alpha,
        curve_fit_kappa,
        curve_fit_tan_lambda,
        ml_alpha,
        ml_kappa,
        ml_tan_lambda,
        extracted_rows,
        curve_fit_chi_sq,
    ) = extract_curve_fit_and_ml_data(
        get_rows_from_indices,
        dataset,
        predicted_params,
        rows_in_dataset,
        test_helix_data,
    )

    # generates hits from the ml parameter predictions
    ml_chi_squared = chi_squared_for_ml(predicted_params, extracted_rows)

    (
        curve_fit_chi_sq,
        ml_chi_squared,
        curve_fit_mean,
        ml_mean,
        curve_fit_std,
        ml_std,
        residual_curve_fit_alpha,
        residual_curve_fit_kappa,
        residual_curve_fit_tan_lambda,
        residual_ml_alpha,
        residual_ml_kappa,
        residual_ml_tan_lambda,
        mse_curve_fit_alpha,
        mse_curve_fit_kappa,
        mse_curve_fit_tan_lambda,
        mse_ml_alpha,
        mse_ml_kappa,
        mse_ml_tan_lambda,
    ) = calc_residuals(
        true_alpha,
        true_kappa,
        true_tan_lambda,
        curve_fit_alpha,
        curve_fit_kappa,
        curve_fit_tan_lambda,
        ml_alpha,
        ml_kappa,
        ml_tan_lambda,
        curve_fit_chi_sq,
        ml_chi_squared,
    )

    write_stats(
        curve_fit_mean,
        ml_mean,
        curve_fit_std,
        ml_std,
        mse_curve_fit_alpha,
        mse_curve_fit_kappa,
        mse_curve_fit_tan_lambda,
        mse_ml_alpha,
        mse_ml_kappa,
        mse_ml_tan_lambda,
        output_file_path=stats_file_path,
    )

    # plot the chi squared
    plot_chi_squared(curve_fit_chi_sq, ml_chi_squared)

    # plotting the residuals
    plot_residuals(
        residual_curve_fit_alpha,
        residual_curve_fit_kappa,
        residual_curve_fit_tan_lambda,
        residual_ml_alpha,
        residual_ml_kappa,
        residual_ml_tan_lambda,
    )
