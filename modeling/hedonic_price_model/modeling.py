#! /usr/bin/env python3
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

'''
Feature engineering
'''


import numpy as np
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
import geopandas as gpd
import pandas as pd
from spreg import OLS, ML_Lag, ML_Error
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.cluster import KMeans
import libpysal
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import logging
import warnings

warnings.filterwarnings("ignore")

def spatial_cv_econometric_models(
        y,
        X,
        coords,
        n_clusters:int=10,
        n_splits:int=5,
        std_dev:bool=False,
        k:int=10
    ) -> pd.DataFrame:
    '''
    Performs spatial cross validation for a given GeoDataFrame and returns
    the follwing performance metrics:
        
        * Root Mean Squared Error (RMSE)
        * Mean Absolute Error (MAE)
        * R²
        * Standard Deviation for RMSE and MAE (optional)
    
    Parameters
    ----------
    y : pd.Series
        target variable in gdf
    X : gpd.GeoDataFrame
        explanatory variables in gdf
    coords: np.ndarray
        array with [lon, lat] for all the records, obtained via
        `np.vstack([gdf.centroid.x, gdf.centroid.y]).T`
    k: int=10
        number of clusters for the data
    splits: int=5
        number of splits for the cross-validation, this would be the k in k-fold
    std_dev: bool=Flase
        include standard deviation for RMSE and MAE if True
    k: int=10
        k for the KNN matrix for SLM and SEM models
    
    Returns
    -------
    pd.DataFrame:
        Dataframe with the models and the metrics
    '''
    y = y.values
    X = X.values
    models = ['OLS', 'SLM', 'SEM']
    rmse = {'OLS':[], 'SLM':[], 'SEM':[]}
    mae = {'OLS':[], 'SLM':[], 'SEM':[]}
    r2 = {'OLS':[], 'SLM':[], 'SEM':[]}

    # get the train and test indexes for cross validation
    blocks = KMeans(n_clusters=n_clusters).fit_predict(coords)  # clusters
    group_kfold = GroupKFold(n_splits=n_splits)  # splits
    train_test_sets = group_kfold.split(X, y, groups=blocks)  # indexes 

    # get metrics for every fold
    for fold, (train_idx, test_idx) in enumerate(train_test_sets):
        
        # get the train and test sets
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        coords_train, coords_test = coords[train_idx], coords[test_idx]
        # create KNN weights for SLM and SEM
        knn_train = libpysal.weights.KNN.from_array(coords_train, k=k)
        knn_train.transform = 'r'  # row-standardize the spatial weigth
        knn_test = libpysal.weights.KNN.from_array(coords_test, k=k)
        knn_test.transform = 'r'  # row-standardize the spatial weigth

        try:
            # fit the models
            ols_model = OLS(y=y_train, x=X_train)
            slm_model = ML_Lag(y=y_train, x=X_train, w=knn_train)
            sem_model = ML_Error(y=y_train, x=X_train, w=knn_train)

            # test models
            # ols
            X_test_ols = np.hstack([np.ones((len(X_test), 1)), X_test])
            y_pred_ols = X_test_ols @ ols_model.betas
            # slm
            rho = slm_model.betas[-1][0]  # Spatial lag parameter
            beta_coefs = slm_model.betas[:-1]
            X_test_slm = np.column_stack([np.ones(len(X_test)), X_test])
            y_pred_slm = np.linalg.inv(np.eye(len(X_test_slm)) - rho * knn_test.full()[0]) @ (X_test_slm @ beta_coefs)
            # sem
            sem_model = ML_Error(y_train, X_train, w=knn_train)
            X_test_sem = np.column_stack([np.ones(len(X_test)), X_test])
            y_pred_sem = X_test_sem @ sem_model.betas[:-1]  # Exclude lambda

            # get metrics
            # rmse
            rmse['OLS'].append(root_mean_squared_error(y_test, y_pred_ols))
            rmse['SLM'].append(root_mean_squared_error(y_test, y_pred_slm))
            rmse['SEM'].append(root_mean_squared_error(y_test, y_pred_sem))
            # mae
            mae['OLS'].append(mean_absolute_error(y_test, y_pred_ols))
            mae['SLM'].append(mean_absolute_error(y_test, y_pred_slm))
            mae['SEM'].append(mean_absolute_error(y_test, y_pred_sem))
            # r2 of training data
            r2['OLS'].append(ols_model.r2)
            r2['SLM'].append(slm_model.pr2)
            r2['SEM'].append(sem_model.pr2)
        
        except Exception as e:
            logging.warning(f'Error in fold {fold+1}: {e}')
        
    # create results dataframe
    res = {'Model': models, 'RMSE':[], 'MAE':[], 'R²':[]}
    for model in models:
        if std_dev:
            res['RMSE'].append(f'{np.mean(rmse[model]):,.2f} (+/- {np.std(rmse[model]):,.2f})')
            res['MAE'].append(f'{np.mean(mae[model]):,.2f} (+/- {np.std(mae[model]):,.2f})')
            res['R²'].append(f'{np.mean(r2[model]):,.4f} (+/- {np.std(r2[model]):,.4f})')
        else:    
            res['RMSE'].append(np.mean(rmse[model]))
            res['MAE'].append(np.mean(mae[model]))
            res['R²'].append(np.mean(r2[model]))
    
    return pd.DataFrame(res)

    
def spatial_cv_xgboost(
        y,
        X,
        coords,
        n_clusters: int = 10,
        n_splits: int = 5,
        std_dev: bool = False,
        k: int = 10
    ) -> pd.DataFrame:
    """
    Performs spatial cross-validation for XGBoost using grouped folds and returns
    RMSE, MAE, and R², optionally including their standard deviations.

    Parameters
    ----------
    y : pd.Series or np.ndarray
        Target variable.
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    coords : np.ndarray
        Array with [lon, lat] for all records.
    n_clusters : int, default=10
        Number of spatial clusters to group observations.
    n_splits : int, default=5
        Number of CV folds.
    std_dev : bool, default=False
        Include standard deviation for RMSE, MAE, and R².
    k : int, default=10
        Number of neighbors for KNN spatial weights.

    Returns
    -------
    pd.DataFrame
        DataFrame with mean (and optional std) metrics for the XGBoost model.
    """

    # Convert input data if needed
    y = np.array(y)
    X = np.array(X)

    # Initialize metrics storage
    metrics = {'RMSE': [], 'MAE': [], 'R²': []}

    # Create spatial clusters and group-based CV splits
    blocks = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(coords)
    group_kfold = GroupKFold(n_splits=n_splits)
    train_test_sets = group_kfold.split(X, y, groups=blocks)

    # Base XGBoost model
    xgb_base = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    # Hyperparameter space
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }

    # RandomizedSearchCV for hyperparameter tuning
    xgb_search = RandomizedSearchCV(
        xgb_base,
        param_distributions=param_dist,
        n_iter=20,
        cv=GroupKFold(n_splits=3),
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1
    )

    # Spatial CV loop
    for fold, (train_idx, test_idx) in enumerate(train_test_sets):
        try:
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            coords_train, coords_test = coords[train_idx], coords[test_idx]

            # Create KNN spatial weights (not used directly, but kept if needed)
            knn_train = libpysal.weights.KNN.from_array(coords_train, k=k)
            knn_train.transform = 'r'
            knn_test = libpysal.weights.KNN.from_array(coords_test, k=k)
            knn_test.transform = 'r'

            # Fit the search object using training data and its groups
            fold_groups = blocks[train_idx]  # important fix!
            xgb_search.fit(X_train, y_train, groups=fold_groups)

            # Best model from tuning
            final_xgb = xgb_search.best_estimator_

            # Predict on test fold
            y_pred = final_xgb.predict(X_test)

            # Compute metrics
            rmse = root_mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            metrics['RMSE'].append(rmse)
            metrics['MAE'].append(mae)
            metrics['R²'].append(r2)

        except Exception as e:
            logging.warning(f"Error in fold {fold + 1}: {e}")

    # Build results DataFrame
    res = {'Model': ['XGBoost']}

    if std_dev:
        res['RMSE'] = [f"{np.mean(metrics['RMSE']):,.2f} (+/- {np.std(metrics['RMSE']):,.2f})"]
        res['MAE'] = [f"{np.mean(metrics['MAE']):,.2f} (+/- {np.std(metrics['MAE']):,.2f})"]
        res['R²'] = [f"{np.mean(metrics['R²']):,.4f} (+/- {np.std(metrics['R²']):,.4f})"]
    else:
        res['RMSE'] = [np.mean(metrics['RMSE'])]
        res['MAE'] = [np.mean(metrics['MAE'])]
        res['R²'] = [np.mean(metrics['R²'])]

    return pd.DataFrame(res)

import numpy as np
import libpysal
from spreg import OLS, ML_Lag, ML_Error
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

def fit_all_models(y, X, coords, k:int=10) -> tuple:
    """
    Fits OLS, SLM, SEM, GWR and XGBoost models on the full dataset.
    
    Parameters
    ----------
    y : pd.Series or np.ndarray
        Target variable.
    X : pd.DataFrame or np.ndarray
        Explanatory variables.
    coords : np.ndarray
        Array with [lon, lat] coordinates for all records.
    k : int, optional (default=10)
        Number of nearest neighbors for the KNN spatial weights matrix.
    
    Returns
    -------
    tuple
        (ols_model, slm_model, sem_model, xgb_model)
    """
    X_xgb = X.copy()
    X_copy = X.drop(columns=['lat', 'lon']).copy()
    # Ensure numpy arrays
    y = np.asarray(y)
    X = np.asarray(X_copy)
    
    # Create spatial weights matrix
    knn = libpysal.weights.KNN.from_array(coords, k=k)
    knn.transform = 'r'

    # --- OLS ---
    ols_model = OLS(y=y, x=X, name_x=list(X_copy.columns))
    
    # --- Spatial Lag Model (SLM) ---
    slm_model = ML_Lag(y=y, x=X, w=knn, name_x=list(X_copy.columns))
    
    # --- Spatial Error Model (SEM) ---
    sem_model = ML_Error(y=y, x=X, w=knn, name_x=list(X_copy.columns))
    
    # --- XGBoost ---
    xgb_base = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Perform hyperparameter tuning using RandomizedSearchCV
    xgb_search = RandomizedSearchCV(
        xgb_base, param_distributions=param_dist,
        n_iter=20, scoring='neg_mean_squared_error',
        cv=3, random_state=42, n_jobs=-1
    )
    xgb_search.fit(X_xgb, y)
    print(xgb_search.best_params_)
    
    # Retrieve best XGBoost model
    xgb_model = xgb_search.best_estimator_

    # # --- GWR Bandwidth ---
    # selector = Sel_BW(coords=coords, y=y, X_loc=X,
    #                   fixed=False, kernel='gaussian')
    # optimal_badwidth = selector.search(criterion='AICc', verbose=False)
    # # --- GWR ---
    # gwr_model = GWR(coords, y, X, bw=optimal_badwidth,
    #                 fixed=False, kernel='gaussian').fit()
    
    return ols_model, slm_model, sem_model, xgb_model