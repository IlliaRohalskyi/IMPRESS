import os
import mlflow
from src.utils import get_project_root, save_pickle
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_absolute_error
import optuna
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
    
class ModelTrainer:
    
    def create_best_model(self, model_name, best_params):
        if model_name == "rf":
            model = RandomForestRegressor(**best_params)
        elif model_name == "adaboost":
            model = AdaBoostRegressor(**best_params)
        elif model_name == "xgb":
            model = XGBRegressor(tree_method='gpu_hist', **best_params)
        elif model_name == "lgb":
            model = LGBMRegressor(device='gpu', **best_params)
        elif model_name == "catboost":
            model = CatBoostRegressor(logging_level='Silent', task_type='GPU', **best_params)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        model.fit(self.X_train, self.y_train)
        
        return model
    
    def objective(self, trial):
        model_name = self.current_model_name
        
        if model_name == "xgb":
            model = XGBRegressor(
                n_estimators=trial.suggest_int('n_estimators', 5, 300),
                max_depth=trial.suggest_int('max_depth', 3, 100),
                learning_rate=trial.suggest_float('learning_rate', 0.001, 0.3),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                gamma=trial.suggest_float('gamma', 0, 5),
                min_child_weight=trial.suggest_int('min_child_weight', 1, 20),
                reg_alpha=trial.suggest_float('reg_alpha', 0, 1),
                reg_lambda=trial.suggest_float('reg_lambda', 0, 1),
            )
        elif model_name == "rf":
            model = RandomForestRegressor(
                n_estimators=trial.suggest_int('n_estimators', 5, 300),
                max_depth=trial.suggest_int('max_depth', 3, 100),
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
            mae = mean_absolute_error(y_val_fold, predictions)
            mae_list.append(mae)
        print('here')
        return np.mean(mae_list)

    
    def initiate_model_training(self, train, test):
        
        self.X_train = train[:, :-3]
        self.y_train = train[:, -3:]
        
        self.X_test = test[:, :-3]
        self.y_test = test[:, -3:] 
        
        models = ["xgb", "rf"]
        
        mlflow.set_tracking_uri('https://dagshub.com/IlliaRohalskyi/IMPRESS.mlflow')

        with mlflow.start_run() as run:
            data_ingestion_script_path = os.path.join(get_project_root(), 'src/components/data_ingestion.py')
            data_transformation_script_path = os.path.join(get_project_root(), 'src/components/data_transformation.py')
            scaler_path = os.path.join(get_project_root(), 'artifacts/data_processing/scaler.pkl')
            mlflow.log_artifact(scaler_path, artifact_path="components")
            mlflow.log_artifact(data_ingestion_script_path, artifact_path="components")
            mlflow.log_artifact(data_transformation_script_path, artifact_path="components")
                    
            best_models = []
            best_params = []
            best_maes = []
            
            for model_name in models:
                self.current_model_name = model_name
                
                sampler = optuna.samplers.TPESampler(seed=42)
                study = optuna.create_study(direction="minimize", sampler=sampler)
                study.optimize(self.objective, n_trials=200, show_progress_bar=True)
                
                best_params.append(study.best_params)
                best_maes.append(study.best_value)
                params_with_prefix = {f"{model_name}_{key}": value for key, value in study.best_params.items()}

                mlflow.log_params(params_with_prefix)
                mlflow.log_metric(f"{model_name}_val_mae", study.best_value)
                
                best_model = self.create_best_model(model_name, study.best_params)
                best_models.append(best_model)
                
            ensemble_predictions = np.zeros_like(self.y_test)
            print(ensemble_predictions.shape)
            total_weight = sum(1/mae for mae in best_maes)
            weights = [1/mae/total_weight for mae in best_maes]
            mlflow.log_params({"weights": weights})
            
            for model, weight in zip(best_models, weights):
                predictions = model.predict(self.X_test)
                ensemble_predictions += weight * predictions
                
            ensemble_mae = mean_absolute_error(self.y_test, ensemble_predictions, multioutput='raw_values')

            print("Ensemble MAE for Oberflaechenspannung:", ensemble_mae[0])
            print("Ensemble MAE for Anionischetenside:", ensemble_mae[1])
            print("Ensemble MAE for Nichtionischentenside:", ensemble_mae[2])

            mlflow.log_metric("ensemble_mae_bberflaechenspannung", ensemble_mae[0])
            mlflow.log_metric("ensemble_mae_anionischetenside", ensemble_mae[1])
            mlflow.log_metric("ensemble_mae_nichtionischentenside", ensemble_mae[2])


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation
    ing = DataIngestion()
    tr = DataTransformation()
    trainer = ModelTrainer()
    off, on = ing.initiate_data_ingestion()
    train, test = tr.initiate_data_transformation(off, on, True)
    trainer.initiate_model_training(train, test)
