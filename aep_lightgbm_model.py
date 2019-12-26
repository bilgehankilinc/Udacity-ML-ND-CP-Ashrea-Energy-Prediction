import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import matplotlib.pyplot as plt

# Random State
np.random.seed(42)

# Pycharm Viewing Arrangement
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 15)

df_X = pd.read_csv(r'preprocessed-data\df_X.csv', compression='zip')

# Defining our numerical features
l_numerical_features = ['square_feet', 'air_temperature', 'dew_temperature', 'sea_level_pressure', 'eui_2016']

# Casting our categorical features
l_categorical_features = ['building_id', 'meter', 'site_id', 'primary_use', 'wind_direction', 'month', 'hour', 'season']

df_X[l_categorical_features] = df_X[l_categorical_features].astype('category')

# Dropping Least Important Features from Feature Importance
l_least_important_features = ['is_weekend', 'is_semester_on', 'dayofweek', 'wind_speed', 'age', 'cloud_coverage',
                              'is_holiday', 'precip_depth_1_hr']

df_X = df_X.drop(l_least_important_features, axis=1)

# Create X, X_given_test and y variables. Also Log Transforming our target variable.
X = df_X.drop('meter_reading', axis=1)
y = np.log1p(df_X['meter_reading'])

# Train and Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=42)
del X, y

# Creating LightGBM Regression Instance
train_lgb = lgb.Dataset(X_train, label=y_train)
test_lgb = lgb.Dataset(X_test, label=y_test, reference=train_lgb)

params = {'task': 'train', 'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
          'num_boost_round': 256, 'early_stopping_rounds': 50, 'verbose': 1,
          'learning_rate': 0.05, 'num_leaves': 256, 'feature_fraction': 0.8, 'bagging_fraction': 0.8}

gbm = lgb.train(params, train_lgb, valid_sets=[train_lgb, test_lgb], valid_names=['train', 'eval'],
                categorical_feature=l_categorical_features)

# Predictions and Metrics
y_pred_train = gbm.predict(X_train)
y_pred_train[y_pred_train < 0] = 0

y_pred = gbm.predict(X_test)
y_pred[y_pred < 0] = 0
df_imp = pd.DataFrame(gbm.feature_importance(), index=X_train.columns,
                      columns=['importance']).sort_values(by='importance', ascending=False)
del X_train, X_test

# Printing Metrics
print('Lightgbm Regression RMSE Train: ', np.sqrt(mean_squared_error(y_train, y_pred_train)))
print('Lightgbm Regression RMSLE Train: ', np.sqrt(mean_squared_log_error(y_train, y_pred_train)))

print('Lightgbm Regression RMSE Test: ', np.sqrt(mean_squared_error(y_test, y_pred)))
print('Lightgbm Regression RMSLE Test: ', np.sqrt(mean_squared_log_error(y_test, y_pred)))

# Plotting Feature Importance
df_imp.plot(kind='bar')
plt.show()

# Kaggle Competition Test Data Prediction and Submission File Creation
df_X_given_test = pd.read_csv(r'preprocessed-data\df_X_given_test.csv', compression='zip')
df_X_given_test[l_categorical_features] = df_X_given_test[l_categorical_features].astype('category')
X_given_test = df_X_given_test.drop('row_id', axis=1)

# Predictions on Competitions Unlabeled Test Data
y_pred_test = gbm.predict(X_given_test)
y_pred_test[y_pred_test < 0] = 0
del X_given_test

# Creating a Kaggle Submission CSV
df_X_given_test['meter_reading'] = y_pred_test
df_submission = df_X_given_test.set_index('row_id')['meter_reading'].round(2)
del df_X_given_test
df_submission.to_csv(r'sample_submission.csv', header=True)
