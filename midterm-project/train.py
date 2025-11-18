# train.py
import pickle

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    roc_auc_score, precision_score, recall_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import warnings

warnings.filterwarnings('ignore')

print(f'pandas =={pd.__version__}')
print(f'numpy =={np.__version__}')
print(f'sklearn=={sklearn.__version__}')


def load_data(file_path= 'water_potability.csv'):
    df = pd.read_csv(file_path)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def prepare_data(df, test_size=0.2, random_state=42):
    X = df.drop('Potability', axis=1)
    y = df['Potability']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"\nTarget distribution in training set:")
    print(y_train.value_counts(normalize=True))
    
    return X_train, X_test, y_train, y_test


def evaluate_pipeline_model(pipeline, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a single pipeline model"""
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, 'predict_proba') else None
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall_not_potable = recall_score(y_test, y_pred, pos_label=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'recall_not_potable': recall_not_potable,
        'roc_auc': roc_auc,
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std()
    }
    
    print(f"\n{model_name}:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}" if roc_auc else "  ROC-AUC:   N/A")
    print(f"  CV F1:     {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return results, pipeline


def train_baseline_models(X_train, X_test, y_train, y_test):
    """Train multiple baseline models wrapped in pipelines"""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Naive Bayes': GaussianNB()
    }
    
    results = []
    trained_pipelines = {}
    
    for name, model in models.items():
        pipeline = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        result, trained_pipeline = evaluate_pipeline_model(
            pipeline, X_train, X_test, y_train, y_test, name
        )
        results.append(result)
        trained_pipelines[name] = trained_pipeline
    
    results_df = pd.DataFrame(results).sort_values('f1_score', ascending=False)
    best_model_name = results_df.iloc[0]['model_name']
    
    return results_df, trained_pipelines, best_model_name


def tune_best_model(X_train, y_train, best_model_name):
    """Perform hyperparameter tuning on the best model"""
    print(f"\n{'='*60}")
    print(f"Tuning: {best_model_name}")
    print(f"{'='*60}")
    
    if best_model_name == 'Random Forest':
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [5, 10, 15, None],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4]
        }
    elif best_model_name == 'Gradient Boosting':
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7],
            'model__min_samples_split': [2, 5, 10]
        }
    elif best_model_name == 'Logistic Regression':
        model = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {
            'model__C': [0.01, 0.1, 1, 10, 100],
            'model__penalty': ['l2'],
            'model__solver': ['lbfgs', 'liblinear']
        }
    else:
        print(f"No tuning implemented for {best_model_name}, using baseline")
        return None
    
    pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def final_evaluation(pipeline, X_test, y_test, model_name="Final Model"):
    """Detailed evaluation of the final model"""
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, 'predict_proba') else None
    
    print(f"\n{'='*60}")
    print(f"{model_name} Performance on Test Set")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
    if y_pred_proba is not None:
        print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Potable', 'Potable']))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Not   Potable")
    print(f"Actual Not    {cm[0,0]:<5} {cm[0,1]:<5}")
    print(f"       Potable {cm[1,0]:<5} {cm[1,1]:<5}")


def save_model(pipeline, feature_names, model_path='model.bin'):
    """Save the trained model and feature names"""
    model_data = {
        'pipeline': pipeline,
        'feature_names': feature_names
    }
    
    with open(model_path, 'wb') as f_out:
        pickle.dump(model_data, f_out)
    
    print(f"\nModel saved to {model_path}")


def main():
    # Load data
    df = load_data('water_potability.csv')
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)
    feature_names = X_train.columns.tolist()
    # Train baseline models
    results_df, trained_models, best_model_name = train_baseline_models(
        X_train, X_test, y_train, y_test
    )

    print("Baseline Model Comparison")
    print(results_df.to_string(index=False))
    
    # Tune best model
    tuned_model = tune_best_model(X_train, y_train, best_model_name)
    
    # Select final model
    final_model = tuned_model if tuned_model else trained_models[best_model_name]
    
    # Final evaluation
    final_evaluation(final_model, X_test, y_test, 
                    "Tuned Model" if tuned_model else "Baseline Model")
    
    # Save model
    save_model(final_model, feature_names)

if __name__ == "__main__":
    main()