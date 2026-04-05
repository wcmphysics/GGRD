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

# Additional utility imports that might be helpful for stacking
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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
        'random_state': 42
    }
    
    lr_hyperparams = {
        'fit_intercept': True,
        'copy_X': True,
        'n_jobs': -1
    }
    
    estimators = [
        ('rf', RandomForestRegressor(**rf_hyperparams)),
        ('lr', LinearRegression(**lr_hyperparams))
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


