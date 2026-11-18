import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

def select_best_model_and_tune(trained_models_dict, X_test, y_test, X_train, y_train):
    best_score = 0
    best_model_name = ""
    best_model_instance = None

    print("--- Step 1: Best Model Selection ---")
    for name, model in trained_models_dict.items():
        preds = model.predict(X_test)
        score = accuracy_score(y_test, preds)
        print(f"Model: {name} | Accuracy: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model_name = name
            best_model_instance = model

    print(f"\n🏆 Selected Best Model: {best_model_name} with Accuracy: {best_score:.4f}")

    print(f"\n--- Step 2: Hyperparameter Tuning for {best_model_name} ---")
    
    param_grids = {
        "Random Forest": {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        },
        "Gradient Boosting": {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5]
        },
        "SVM": {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        },
        "KNN": {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        },
        "Logistic Regression": {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear']
        }
    }

    grid = param_grids.get(best_model_name, {})
    
    grid_search = GridSearchCV(
        estimator=best_model_instance, 
        param_grid=grid, 
        cv=3, 
        scoring='accuracy', 
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    print("Best Hyperparameters Found:", grid_search.best_params_)
    
    final_tuned_model = grid_search.best_estimator_
    final_preds = final_tuned_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, final_preds)
    
    print(f"Final Tuned Model Accuracy: {final_accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, final_preds))
    
    return final_tuned_model

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    print("Running local independent test of your module...")
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mock_models = {
        "Random Forest": RandomForestClassifier(random_state=42).fit(X_train, y_train),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42).fit(X_train, y_train)
    }

    select_best_model_and_tune(mock_models, X_test, y_test, X_train, y_train)