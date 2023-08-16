import os
import subprocess
import mlflow
from dataclasses import dataclass
from src.utils import get_project_root, save_pickle, load_pickle
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_absolute_error
import optuna
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

@dataclass
class ModelTrainerConfig:
    data_ingestion_script_path = os.path.join(get_project_root(),
                                                'src/components/data_ingestion.py')

    data_transformation_script_path = os.path.join(get_project_root(),
                                                    'src/components/data_transformation.py')

    feature_scaler_path = os.path.join(get_project_root(),
                                        'artifacts/data_processing/feature_scaler.pkl')

    target_scaler_path = os.path.join(get_project_root(),
                                        'artifacts/data_processing/target_scaler.pkl')
    
    model_path = os.path.join(get_project_root(), 'artifacts/models')
    
    explainability_path = os.path.join(get_project_root(), 'artifacts/explainability')
    
class ModelTrainer:
    
    def __init__(self):
        self.git_hash = self._get_git_hash()
        self.trainer_config = ModelTrainerConfig()
        
    def train_model(self, X_train, y_train, model_name, best_params):
        if model_name == "rf":
            model = RandomForestRegressor(**best_params)
        elif model_name == "xgb":
            model = XGBRegressor(tree_method='gpu_hist', **best_params)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        model.fit(X_train, y_train)
        
        return model
    
    def objective(self, trial):
        model_name = self.current_model_name
        
        if model_name == "xgb":
            model = XGBRegressor(
                n_estimators=trial.suggest_int('n_estimators', 5, 30000),
                max_depth=trial.suggest_int('max_depth', 3, 30000),
                learning_rate=trial.suggest_float('learning_rate', 0.001, 0.3),
                subsample=trial.suggest_float('subsample', 0.1, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.1, 1.0),
                gamma=trial.suggest_float('gamma', 0, 10),
                min_child_weight=trial.suggest_int('min_child_weight', 1, 20),
                reg_alpha=trial.suggest_float('reg_alpha', 0, 1),
                reg_lambda=trial.suggest_float('reg_lambda', 0, 1),
            )
        elif model_name == "rf":
            model = RandomForestRegressor(
                n_estimators=trial.suggest_int('n_estimators', 5, 30000),
                max_depth=trial.suggest_int('max_depth', 3, 30000),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 30),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 15),
                max_features=trial.suggest_float('max_features', 0.1, 1),
                bootstrap=trial.suggest_categorical('bootstrap', [True, False]),
            )

        
        mae_list = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(self.X_train, self.y_train):
            X_train_fold, X_val_fold = self.X_train[train_idx], self.X_train[val_idx]
            y_train_fold, y_val_fold = self.y_train[train_idx], self.y_train[val_idx]

            model.fit(X_train_fold, y_train_fold)
            predictions = model.predict(X_val_fold)
            mae = self.target_scaler.inverse_transform(mean_absolute_error(y_val_fold, predictions, multioutput='raw_values').reshape(1, -1))
            
            average_scaled_mae = np.mean(mae)
            
            mae_list.append(average_scaled_mae)

        return np.mean(mae_list)
    
    def _get_git_hash(self):
        try:
            git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
            return git_hash
        except subprocess.CalledProcessError as e:
            print(f"Error: Failed to retrieve Git hash. Command returned {e.returncode}: {e.output}")
            return "Unknown"

    def initiate_model_training(self, X_train, y_train, X_test, y_test, feature_names=None):
        
        self.X_train = X_train
        self.y_train = y_train
        
        self.X_test = X_test
        self.y_test = y_test
        
        self.feature_names = feature_names
        
        self.target_scaler = load_pickle(self.trainer_config.target_scaler_path)
        
        models = ["xgb", "rf"]
        
        mlflow.set_tracking_uri('https://dagshub.com/IlliaRohalskyi/IMPRESS.mlflow')

        with mlflow.start_run() as run:
            
            mlflow.log_param("git_hash", self.git_hash)

            mlflow.log_artifact(self.trainer_config.feature_scaler_path,
                                artifact_path="scalers")

            mlflow.log_artifact(self.trainer_config.target_scaler_path,
                                artifact_path="scalers")

            mlflow.log_artifact(self.trainer_config.data_ingestion_script_path,
                                artifact_path="components")

            mlflow.log_artifact(self.trainer_config.data_transformation_script_path,
                                artifact_path="components")
                    
            best_models = []
            best_params = []
            best_maes = []
            
            for model_name in models:
                self.current_model_name = model_name
                
                sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=35)
                study = optuna.create_study(direction="minimize", sampler=sampler)
                study.optimize(self.objective, n_trials=100, show_progress_bar=True)
                
                best_params.append(study.best_params)
                best_maes.append(study.best_value)
                params_with_prefix = {f"{model_name}_{key}": value for key, value in study.best_params.items()}

                mlflow.log_params(params_with_prefix)
                mlflow.log_metric(f"{model_name}_val_total_mae", study.best_value)
                
                best_model = self.train_model(self.X_train, self.y_train, model_name, study.best_params)
                best_models.append(best_model)
                
                os.makedirs(self.trainer_config.model_path, exist_ok=True)
                model_path = os.path.join(self.trainer_config.model_path, f'{model_name}.pkl')
                save_pickle(best_model, model_path)
                mlflow.log_artifact(model_path, artifact_path='models')

                preds = best_model.predict(self.X_test)
                mae = self.target_scaler.inverse_transform(mean_absolute_error(self.y_test, preds, multioutput='raw_values').reshape(1, -1)).flatten()

                mlflow.log_metric(f"{model_name}_mae_oberflaechenspannung", mae[0])
                mlflow.log_metric(f"{model_name}_mae_anionischetenside", mae[1])
                mlflow.log_metric(f"{model_name}_mae_nichtionischentenside", mae[2])
                mlflow.log_metric(f"{model_name}_mae_total", np.mean(mae))
                
                if self.feature_names is not None:
                    self.feature_importance_plot(best_model, model_name)
                
                mlflow.log_artifacts(self.trainer_config.explainability_path, artifact_path='explainability')
            
            ensemble_predictions = np.zeros_like(self.y_test)
            total_weight = sum(1/mae for mae in best_maes)
            weights = [1/mae/total_weight for mae in best_maes]
            mlflow.log_params({"weights": weights})
            
            for model, weight in zip(best_models, weights):
                predictions = model.predict(self.X_test)
                ensemble_predictions += weight * predictions
                
            ensemble_mae = self.target_scaler.inverse_transform(mean_absolute_error(self.y_test, ensemble_predictions, multioutput='raw_values').reshape(1, -1)).flatten()

            mlflow.log_metric("ensemble_mae_oberflaechenspannung", ensemble_mae[0])
            mlflow.log_metric("ensemble_mae_anionischetenside", ensemble_mae[1])
            mlflow.log_metric("ensemble_mae_nichtionischentenside", ensemble_mae[2])
            mlflow.log_metric("ensemble_mae_total", np.mean(ensemble_mae))
                
    def feature_importance_plot(self, model, model_name):
        feature_importances = model.feature_importances_
        n_features = len(feature_importances)
        plt.figure(figsize=(10, 6))
        plt.barh(range(n_features), feature_importances, align="center")
        plt.yticks(np.arange(n_features), self.feature_names)
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        plt.title(f"{model_name} Model - Feature Importance")
        plt.tight_layout()
        os.makedirs(self.trainer_config.explainability_path, exist_ok=True)
        plt.savefig(os.path.join(self.trainer_config.explainability_path, f"{model_name}_feature_importance.png"))
        plt.close()

if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation
    ing = DataIngestion()
    tr = DataTransformation()
    trainer = ModelTrainer()
    off, on = ing.initiate_data_ingestion()
    x_train, y_train, x_test, y_test, feature_names = tr.initiate_data_transformation(off, on)
    trainer.initiate_model_training(x_train, y_train, x_test, y_test, feature_names=feature_names)
