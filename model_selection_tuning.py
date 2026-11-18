import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

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
            n_estimators = trial.suggest_int('n_estimators', 50, 150)
            max_depth = trial.suggest_categorical('max_depth', [10, 20])
            min_samples_split = trial.suggest_int('min_samples_split', 2, 8)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
        
        elif best_model_name == "Gradient Boosting":
            n_estimators = trial.suggest_int('n_estimators', 50, 100)
            learning_rate = trial.suggest_float('learning_rate', 0.05, 0.1)
            max_depth = trial.suggest_int('max_depth', 3, 5)
            model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
        else:
            model = best_model_instance

        # Using cv=2 and a smaller sample for extreme speed during tuning
        scores = cross_val_score(model, X_train, y_train, cv=2, scoring='accuracy')
        return scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    # Reduced trials to 3 for instant execution
    study.optimize(objective, n_trials=3)

    print("Best Hyperparameters Found:", study.best_params)
    
    if best_model_name == "Random Forest":
        final_tuned_model = RandomForestClassifier(**study.best_params, random_state=42)
    elif best_model_name == "Gradient Boosting":
        final_tuned_model = GradientBoostingClassifier(**study.best_params, random_state=42)
    else:
        final_tuned_model = best_model_instance

    final_tuned_model.fit(X_train, y_train)
    final_preds = final_tuned_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, final_preds)
    
    print(f"Final Tuned Model Accuracy: {final_accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, final_preds))
    
    return final_tuned_model

if __name__ == "__main__":
    print("Loading team dataset from data/ folder...")
    try:
        df = pd.read_parquet("data/features_train.parquet")
        
        # Subsample dataset to 2,000 rows max so it runs instantly!
        if len(df) > 2000:
            df = df.sample(n=2000, random_state=42)
            print("Dataset subsampled to 2,000 rows for fast execution.")
            
        target_col = df.columns[-1] if 'label' not in df.columns else 'label'
        
        X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
        y = df[target_col].astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"Data loaded successfully. Shape: {X.shape}")

        rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
        gb = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)

        trained_models = {
            "Random Forest": rf,
            "Gradient Boosting": gb
        }

        select_best_model_and_tune(trained_models, X_test, y_test, X_train, y_train)

    except Exception as e:
        print(f"Error loading parquet ({e}). Falling back to mock data...")
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        mock_models = {
            "Random Forest": RandomForestClassifier(random_state=42).fit(X_train, y_train),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42).fit(X_train, y_train)
        }
        select_best_model_and_tune(mock_models, X_test, y_test, X_train, y_train)