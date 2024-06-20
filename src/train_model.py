# Code pour entraîner notre modèle
import mlflow
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from data_preparation import preprocess_data, load_data, split_data

def train_model(X_train, y_train):
    mlflow.start_run(run_name="Model Training")
    
    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=10)
    pipeline = Pipeline([
        ('preprocessor', preprocess_data(X_train)),
        ('classifier', model)
    ])
    
    param_grid = {
        'classifier__max_depth': [3, 5],
        'classifier__n_estimators': [100, 200]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', verbose=2)
    grid_search.fit(X_train, y_train)
    
    mlflow.log_params(grid_search.best_params_)
    mlflow.sklearn.log_model(grid_search.best_estimator_, "model")
    
    mlflow.end_run()

if __name__ == "__main__":
    data = load_data('../data/processed/processed_data_train.csv')
    X_train, _, y_train, _ = split_data(data, 'TARGET')
    train_model(X_train, y_train)
