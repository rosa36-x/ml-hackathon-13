import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

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

    print(f"\n--- Step 2: Bayesian Hyperparameter Tuning for {best_model_name} (Using Optuna) ---")
    
    def objective(trial):
        if best_model_name == "Random Forest":
            from sklearn.ensemble import RandomForestClassifier
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_categorical('max_depth', [None, 10, 20, 30])
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
        
        elif best_model_name == "Gradient Boosting":
            from sklearn.ensemble import GradientBoostingClassifier
            n_estimators = trial.suggest_int('n_estimators', 50, 150)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
            max_depth = trial.suggest_int('max_depth', 3, 6)
            model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
        
        elif best_model_name == "SVM":
            from sklearn.svm import SVC
            C = trial.suggest_float('C', 0.1, 10.0, log=True)
            kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
            model = SVC(C=C, kernel=kernel, random_state=42)
        
        elif best_model_name == "KNN":
            from sklearn.neighbors import KNeighborsClassifier
            n_neighbors = trial.suggest_int('n_neighbors', 3, 11)
            weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        
        elif best_model_name == "Logistic Regression":
            from sklearn.linear_model import LogisticRegression
            C = trial.suggest_float('C', 0.01, 10.0, log=True)
            solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear'])
            model = LogisticRegression(C=C, solver=solver, random_state=42)
        else:
            model = best_model_instance

        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
        return scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)

    print("Best Hyperparameters Found:", study.best_params)
    
    if best_model_name == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        final_tuned_model = RandomForestClassifier(**study.best_params, random_state=42)
    elif best_model_name == "Gradient Boosting":
        from sklearn.ensemble import GradientBoostingClassifier
        final_tuned_model = GradientBoostingClassifier(**study.best_params, random_state=42)
    elif best_model_name == "SVM":
        from sklearn.svm import SVC
        final_tuned_model = SVC(**study.best_params, random_state=42)
    elif best_model_name == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        final_tuned_model = KNeighborsClassifier(**study.best_params)
    elif best_model_name == "Logistic Regression":
        from sklearn.linear_model import LogisticRegression
        final_tuned_model = LogisticRegression(**study.best_params, random_state=42)
    else:
        final_tuned_model = best_model_instance

    final_tuned_model.fit(X_train, y_train)
    final_preds = final_tuned_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, final_preds)
    
    print(f"Final Tuned Model Accuracy: {final_accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, final_preds))
    
    return final_tuned_model

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    print("Running local test with Bayesian Optimization...")
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mock_models = {
        "Random Forest": RandomForestClassifier(random_state=42).fit(X_train, y_train),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42).fit(X_train, y_train)
    }

    select_best_model_and_tune(mock_models, X_test, y_test, X_train, y_train)