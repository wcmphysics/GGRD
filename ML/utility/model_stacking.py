# Import base machine learning models from scikit-learn
# Regressors
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cross_decomposition import PLSRegression


# Stacking specific modules
from sklearn.ensemble import StackingRegressor, StackingClassifier
from xgboost import XGBRegressor

# Additional utility imports that might be helpful for stacking
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_basic_stacking_model():
    """
    Creates a basic Stacking Regressor combining Linear Regression and Random Forest.
    You can modify the hyperparameters of the base models or the meta-model here.
    """
    
    # 1. Define the base estimators
    # Here you can easily modify the hyperparameters for each model
    rf_hyperparams = {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'random_state': 42, 
        'criterion': 'absolute_error',
        'n_jobs': -1
    }
    
    lr_hyperparams = {
        'fit_intercept': True,
        'copy_X': True,
        'n_jobs': -1
    }
    
    xgb_hyperparams = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'n_jobs': -1
    }

    elastic_net_hyperparams = {
        'alpha': 1.0,
        'l1_ratio': 0.5,
        'random_state': 42
    }

    # Example: Define which feature indices (or column names if using pandas DataFrames) each model should use.
    # Replace these lists with your actual feature names or indices.
    features_rf = [0, 1, 2, 3, 4] 
    features_lr = [5, 6, 7, 8, 9]
    features_xgb = [0, 2, 4, 6, 8]
    features_en = [1, 3, 5, 7, 9]

    # Create a pipeline for each model that selects its specific features first
    rf_pipeline = Pipeline([
        ('selector', ColumnTransformer([('select', 'passthrough', features_rf)], remainder='drop')),
        ('model', RandomForestRegressor(**rf_hyperparams))
    ])

    lr_pipeline = Pipeline([
        ('selector', ColumnTransformer([('select', 'passthrough', features_lr)], remainder='drop')),
        ('model', LinearRegression(**lr_hyperparams))
    ])

    xgb_pipeline = Pipeline([
        ('selector', ColumnTransformer([('select', 'passthrough', features_xgb)], remainder='drop')),
        ('model', XGBRegressor(**xgb_hyperparams))
    ])

    en_pipeline = Pipeline([
        ('selector', ColumnTransformer([('select', 'passthrough', features_en)], remainder='drop')),
        ('model', ElasticNet(**elastic_net_hyperparams))
    ])
    
    estimators = [
        ('rf_pipe', rf_pipeline),
        ('lr_pipe', lr_pipeline),
        ('xgb_pipe', xgb_pipeline),
        ('en_pipe', en_pipeline)
    ]
    
    # 2. Define the final meta-model (Blender)
    # This model learns how to best combine the predictions of the base estimators
    meta_model = Ridge(alpha=1.0)
    
    # 3. Create the Stacking Regressor
    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=meta_model,
        cv=5, # Number of cross-validation folds for the meta-model training
        n_jobs=-1
    )
    
    return stacking_regressor

if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import numpy as np
    
    # Generate some dummy data for testing
    X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Initializing stacking model...")
    model = create_basic_stacking_model()
    
    print("Training the stacking model...")
    model.fit(X_train, y_train)
    
    print("Making predictions...")
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    print(f"Test RMSE: {rmse:.4f}")

    # Save the model
    import joblib
    model_filename = "stacking_model.pkl"
    print(f"Saving model to {model_filename}...")
    joblib.dump(model, model_filename)
    print("Model saved successfully. You can load it later using joblib.load()")

    


