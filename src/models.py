"""
Model definitions and factory for tic episode prediction.
Supports multiple model types: Linear Regression, Decision Trees, Random Forest, XGBoost, LightGBM.
"""

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Any, Optional


class ModelFactory:
    """
    Factory for creating different types of models with specified hyperparameters.
    """

    @staticmethod
    def get_model(model_name: str, task_type: str, params: Optional[Dict[str, Any]] = None):
        """
        Get a model instance based on name and task type.

        Parameters
        ----------
        model_name : str
            One of: 'linear', 'ridge', 'lasso', 'decision_tree', 'random_forest', 'xgboost', 'lightgbm'
        task_type : str
            One of: 'regression', 'classification'
        params : dict, optional
            Hyperparameters for the model

        Returns
        -------
        model
            Instantiated scikit-learn compatible model
        """
        if params is None:
            params = {}

        # Regression models
        if task_type == 'regression':
            if model_name == 'linear':
                return LinearRegression(**params)

            elif model_name == 'ridge':
                default_params = {'alpha': 1.0, 'random_state': 42}
                default_params.update(params)
                return Ridge(**default_params)

            elif model_name == 'lasso':
                default_params = {'alpha': 1.0, 'random_state': 42}
                default_params.update(params)
                return Lasso(**default_params)

            elif model_name == 'decision_tree':
                default_params = {'max_depth': 10, 'random_state': 42}
                default_params.update(params)
                return DecisionTreeRegressor(**default_params)

            elif model_name == 'random_forest':
                default_params = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1}
                default_params.update(params)
                return RandomForestRegressor(**default_params)

            elif model_name == 'xgboost':
                default_params = {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'n_jobs': -1
                }
                default_params.update(params)
                return xgb.XGBRegressor(**default_params)

            elif model_name == 'lightgbm':
                default_params = {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbose': -1
                }
                default_params.update(params)
                return lgb.LGBMRegressor(**default_params)

            else:
                raise ValueError(f"Unknown regression model: {model_name}")

        # Classification models
        elif task_type == 'classification':
            if model_name == 'logistic':
                default_params = {'max_iter': 1000, 'random_state': 42}
                default_params.update(params)
                return LogisticRegression(**default_params)

            elif model_name == 'decision_tree':
                default_params = {'max_depth': 10, 'random_state': 42}
                default_params.update(params)
                return DecisionTreeClassifier(**default_params)

            elif model_name == 'random_forest':
                default_params = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1}
                default_params.update(params)
                return RandomForestClassifier(**default_params)

            elif model_name == 'xgboost':
                default_params = {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'n_jobs': -1
                }
                default_params.update(params)
                return xgb.XGBClassifier(**default_params)

            elif model_name == 'lightgbm':
                default_params = {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'n_jobs': -1,
                    'verbose': -1
                }
                default_params.update(params)
                return lgb.LGBMClassifier(**default_params)

            else:
                raise ValueError(f"Unknown classification model: {model_name}")

        else:
            raise ValueError(f"Unknown task type: {task_type}")

    @staticmethod
    def get_default_params(model_name: str, task_type: str) -> Dict[str, Any]:
        """
        Get default parameters for a model.

        Parameters
        ----------
        model_name : str
            Model name
        task_type : str
            Task type (regression or classification)

        Returns
        -------
        dict
            Default parameters
        """
        # This returns empty dict - defaults are set in get_model()
        # But we can use this to get baseline params for hyperparameter search
        if model_name in ['linear', 'logistic']:
            return {}

        elif model_name in ['ridge', 'lasso']:
            return {'alpha': 1.0}

        elif model_name == 'decision_tree':
            return {'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1}

        elif model_name == 'random_forest':
            return {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt' if task_type == 'classification' else 1.0
            }

        elif model_name in ['xgboost', 'lightgbm']:
            return {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 1.0,
                'colsample_bytree': 1.0,
            }

        return {}

    @staticmethod
    def get_param_grid(model_name: str, search_type: str = 'random') -> Dict[str, list]:
        """
        Get hyperparameter search grid for a model.

        Parameters
        ----------
        model_name : str
            Model name
        search_type : str
            'random' for RandomizedSearchCV or 'grid' for GridSearchCV

        Returns
        -------
        dict
            Parameter grid
        """
        if model_name == 'ridge':
            if search_type == 'random':
                return {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
                }
            else:
                return {
                    'alpha': [0.1, 1.0, 10.0]
                }

        elif model_name == 'lasso':
            if search_type == 'random':
                return {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
                }
            else:
                return {
                    'alpha': [0.01, 0.1, 1.0]
                }

        elif model_name == 'decision_tree':
            if search_type == 'random':
                return {
                    'max_depth': [3, 5, 10, 15, 20, None],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 4, 8],
                }
            else:
                return {
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 10],
                    'min_samples_leaf': [1, 4],
                }

        elif model_name == 'random_forest':
            if search_type == 'random':
                return {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [5, 10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', 0.5, 1.0],
                }
            else:
                return {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 15, 20],
                    'min_samples_split': [2, 5],
                    'max_features': ['sqrt', 1.0],
                }

        elif model_name == 'xgboost':
            if search_type == 'random':
                return {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 5, 7, 10],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'min_child_weight': [1, 3, 5],
                }
            else:
                return {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 7],
                    'learning_rate': [0.05, 0.1],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0],
                }

        elif model_name == 'lightgbm':
            if search_type == 'random':
                return {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 5, 7, 10, -1],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'num_leaves': [15, 31, 63, 127],
                    'min_child_samples': [5, 10, 20],
                }
            else:
                return {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 7],
                    'learning_rate': [0.05, 0.1],
                    'num_leaves': [31, 63],
                }

        elif model_name == 'logistic':
            if search_type == 'random':
                return {
                    'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga'],
                }
            else:
                return {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l2'],
                }

        return {}

    @staticmethod
    def get_available_models() -> Dict[str, list]:
        """
        Get list of available models by task type.

        Returns
        -------
        dict
            Dictionary mapping task types to available models
        """
        return {
            'regression': ['linear', 'ridge', 'lasso', 'decision_tree', 'random_forest', 'xgboost', 'lightgbm'],
            'classification': ['logistic', 'decision_tree', 'random_forest', 'xgboost', 'lightgbm'],
        }


if __name__ == "__main__":
    # Test model factory
    print("Testing Model Factory...")

    factory = ModelFactory()

    print("\nAvailable models:")
    available = factory.get_available_models()
    for task, models in available.items():
        print(f"  {task}: {', '.join(models)}")

    print("\nCreating sample models:")

    # Regression models
    print("\nRegression models:")
    rf_reg = factory.get_model('random_forest', 'regression', {'n_estimators': 50, 'max_depth': 5})
    print(f"  Random Forest Regressor: {rf_reg}")

    xgb_reg = factory.get_model('xgboost', 'regression', {'learning_rate': 0.05})
    print(f"  XGBoost Regressor: {xgb_reg}")

    # Classification models
    print("\nClassification models:")
    rf_clf = factory.get_model('random_forest', 'classification', {'n_estimators': 100})
    print(f"  Random Forest Classifier: {rf_clf}")

    lgb_clf = factory.get_model('lightgbm', 'classification')
    print(f"  LightGBM Classifier: {lgb_clf}")

    print("\nSample parameter grids (random search):")
    print(f"  Random Forest: {factory.get_param_grid('random_forest', 'random')}")
    print(f"  XGBoost: {factory.get_param_grid('xgboost', 'random')}")
