import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error

# Random State
np.random.seed(42)

# Pycharm Viewing Arrangement
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 15)

df_X = pd.read_csv(r'preprocessed-data\df_X.csv', compression='zip')

# Defining our numerical features
l_numerical_features = ['square_feet', 'air_temperature', 'dew_temperature', 'precip_depth_1_hr',
                        'sea_level_pressure', 'wind_speed', 'eui_2016', 'age']

# Casting our categorical features
l_categorical_features = ['building_id', 'meter', 'site_id', 'primary_use', 'cloud_coverage', 'wind_direction',
                          'month', 'dayofweek', 'hour', 'is_holiday', 'is_weekend', 'is_semester_on', 'season']

df_X[l_categorical_features] = df_X[l_categorical_features].astype('category')

# Create X, X_given_test and y variables
X = df_X.drop('meter_reading', axis=1)
y = df_X['meter_reading']
del df_X

# Train and Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)
del X, y

# Creating Regressor Instances
d_reg_instances = {'RF': RandomForestRegressor(max_depth=6), 'LGBM': LGBMRegressor()}
df_importance = pd.DataFrame()
df_importance['features'] = X_train.columns

# Calculating RMSLE values for each instance and feature importance
for name, instance in d_reg_instances.items():

    regressor = instance
    regressor.fit(X_train, y_train)

    y_pred_train = regressor.predict(X_train)
    y_pred_train[y_pred_train < 0] = 0
    y_pred_test = regressor.predict(X_test)
    y_pred_test[y_pred_test < 0] = 0

    print('{} RMSLE Train: '.format(name), np.sqrt(mean_squared_log_error(y_train, y_pred_train)))
    print('{} RMSLE Test: '.format(name), np.sqrt(mean_squared_log_error(y_test, y_pred_test)))
    print('\n')
    l_sorter_indices = regressor.feature_importances_.argsort()
    d_importance = dict(zip(X_train.columns[l_sorter_indices],
                            regressor.feature_importances_[l_sorter_indices]/sum(regressor.feature_importances_)))
    df_importance[name] = df_importance['features'].map(d_importance)
    del d_importance, regressor

# Creating Average Importance Values of RF and LGM Models, Sorting
df_importance['Average_Importance'] = (df_importance['RF'] + df_importance['LGBM'])/2
df_importance.set_index('features', inplace=True)
df_importance.sort_values(by='Average_Importance', ascending=False, inplace=True)

# Plotting Cumulative Importance Line Graph
df_importance['Average_Importance'].cumsum().plot(kind='line', drawstyle='steps')
plt.title('RF and LGM Average Importance Cumulative Line Plot')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

# Saving to CSV for Further Use
df_importance.to_csv(r'df_importance.csv')
print(df_importance)
