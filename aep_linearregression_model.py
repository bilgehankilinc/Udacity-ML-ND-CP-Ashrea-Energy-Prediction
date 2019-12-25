import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_squared_log_error

# Random State
np.random.seed(42)

# Pycharm Viewing Arrangement
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 15)

df_X = pd.read_csv(r'preprocessed-data\df_X.csv', compression='zip')

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
X['building_id'] = X['building_id'].map(df_X.groupby('building_id', )['meter_reading'].mean())
X = pd.get_dummies(X, drop_first=True)
X['building_id'] = X['building_id'].astype('category')
del df_X

# Scaling with Standard Scalar
scaler = StandardScaler()
X_scaled = X
for column in X.columns:
    if column in l_numerical_features:
        X_scaled[column] = scaler.fit_transform(X_scaled[column].values.reshape(-1, 1))

# Train and Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, shuffle=True)
del X, y,  X_scaled

# Creating Instances and Making Predictions
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred_train = regressor.predict(X_train)
y_pred_train[y_pred_train < 0] = 0

y_pred = regressor.predict(X_test)
y_pred[y_pred < 0] = 0

print('Linear Regression RMSE Train: ', np.sqrt(mean_squared_error(y_train, y_pred_train)))
print('Linear Regression RMSLE Train: ', np.sqrt(mean_squared_log_error(y_train, y_pred_train)))

print('Linear Regression RMSE Test: ', np.sqrt(mean_squared_error(y_test, y_pred)))
print('Linear Regression RMSLE Test: ', np.sqrt(mean_squared_log_error(y_test, y_pred)))