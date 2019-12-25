import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_squared_log_error

# Random State
np.random.seed(42)

# Pycharm Viewing Arrangement
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 15)

df_X = pd.read_csv(r'preprocessed-data\df_X.csv', compression='zip')
df_X = df_X.sample(n=2000000)

# Defining our numerical features
l_numerical_features = ['square_feet', 'air_temperature', 'dew_temperature', 'sea_level_pressure', 'eui_2016']

# Casting our categorical features
l_categorical_features = ['meter', 'site_id', 'primary_use', 'wind_direction', 'month', 'hour', 'season']

df_X[l_categorical_features] = df_X[l_categorical_features].astype('category')

# Dropping Least Important Features from Feature Importance
l_least_important_features = ['is_weekend', 'is_semester_on', 'dayofweek', 'wind_speed', 'age', 'cloud_coverage',
                              'is_holiday', 'precip_depth_1_hr']

df_X = df_X.drop(l_least_important_features, axis=1)

# Create X, X_given_test and y variables
X = df_X.drop('meter_reading', axis=1)
y = df_X['meter_reading']

# Categorical Encoding with WoE Encoding building_id and dummy encoding other categorical features
X_encoded = X
X_encoded['building_id'] = X_encoded['building_id'].map(df_X.groupby('building_id')['meter_reading'].mean())
X_encoded = pd.get_dummies(X_encoded, drop_first=True)
X_encoded['building_id'] = X_encoded['building_id'].astype('category')
del df_X

# Scaling with Standard Scalar
scaler = StandardScaler()
X_scaled = X_encoded
for column in X.columns:
    if column in l_numerical_features:
        X_scaled[column] = scaler.fit_transform(X_scaled[column].values.reshape(-1, 1))

# Train and Test Split
# LightGBM Regressor uses unencoded and unscaled data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)
# XGBoost Regressor uses encoded but unscaled data.
X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = train_test_split(X_encoded.values, y,
                                                                                    test_size=0.25, shuffle=True)
# Linear and MLP Regressors use encoded and scaled data.
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y, test_size=0.25,
                                                                                shuffle=True)
del X, y, X_encoded, X_scaled

# Defining Regressor Instances
d_regressor_instances = {'Linear Regression': LinearRegression(),
                         'LightGBM Regression': LGBMRegressor(),
                         'XGBoost Regression': XGBRegressor(),
                         'MLP Regression': MLPRegressor(hidden_layer_sizes=(64, 32, 16))}

# Defining a dict to store results
d_model_results = {}

# Creating Instances and Making Predictions
for name, instance in d_regressor_instances.items():
    regressor = instance
    if name == 'LightGBM Regression':
        regressor.fit(X_train, y_train)
    elif name == 'XGBoost Regression':
        regressor.fit(X_train_encoded, y_train_encoded)
    else:
        regressor.fit(X_train_scaled, y_train_scaled)

    # Predictions and Metrics
    if name == 'LightGBM Regression':
        y_pred = regressor.predict(X_test)
    elif name == 'XGBoost Regression':
        y_pred = regressor.predict(X_test_encoded)
    else:
        y_pred = regressor.predict(X_test_scaled)

    y_pred[y_pred < 0] = 0

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    d_model_results[name] = [rmse, rmsle]
    del regressor, rmse, rmsle

del X_train, y_train, X_train_encoded, y_train_encoded, X_train_scaled, y_train_scaled
del X_test, X_test_encoded, X_test_scaled

# Creating a Result DF and Saving it to a CSV
df_model_results = pd.DataFrame(d_model_results, index=['RMSE', 'RMSLE'])
df_model_results.to_csv(r'df_model_selection.csv')
print(df_model_results)
