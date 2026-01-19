import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, confusion_matrix
import joblib
import os

def train_and_evaluate_classification_model():
    """
    Train and evaluate classification model
    """
    print("Training classification model...")
    
    # Read training data
    df = pd.read_csv("comprehensive_classification_data.csv")
    
    # Create features and labels
    X = df.drop(["needs_maintenance"], axis=1)  # Features
    y = df["needs_maintenance"]  # Labels (0: no maintenance needed, 1: maintenance needed)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probability of positive class
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of classification model: {accuracy:.3f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Calculate important metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)  # True Negative Rate
    sensitivity = tp / (tp + fn)  # Recall
    print(f"\nSpecificity (TNR): {specificity:.3f}")
    print(f"Sensitivity (Recall): {sensitivity:.3f}")
    
    # Create directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save model
    joblib.dump(model, "models/classification_model.pkl")
    print("\nClassification model saved to 'models/classification_model.pkl'")
    
    return model, X_test, y_test, y_pred, y_pred_proba

def train_and_evaluate_regression_model():
    """
    Train and evaluate regression model
    """
    print("\nTraining regression model...")
    
    # Read training data
    df = pd.read_csv("comprehensive_regression_data.csv")
    
    # Create features and labels
    X = df.drop(["remaining_km"], axis=1)  # Features
    y = df["remaining_km"]  # Labels (remaining km)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))
    
    print(f"MSE of regression model: {mse:.3f}")
    print(f"RMSE of regression model: {rmse:.3f}")
    print(f"MAE of regression model: {mae:.3f}")
    print(f"R² of regression model: {r2:.3f}")
    
    # Save model
    joblib.dump(model, "models/regression_model.pkl")
    print("\nRegression model saved to 'models/regression_model.pkl'")
    
    return model, X_test, y_test, y_pred

def visualize_results(clf_model, clf_X_test, clf_y_test, clf_y_pred, reg_model, reg_X_test, reg_y_test, reg_y_pred):
    """
    Visualize model results (only if matplotlib is available)
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        
        # Create figure for visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion matrix for classification model
        cm = confusion_matrix(clf_y_test, clf_y_pred)
        im = axes[0,0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[0,0].figure.colorbar(im, ax=axes[0,0])
        axes[0,0].set(xticks=[0, 1], yticks=[0, 1], 
                      xticklabels=['No Maintenance', 'Need Maintenance'],
                      yticklabels=['No Maintenance', 'Need Maintenance'],
                      title='Confusion Matrix - Classification Model',
                      xlabel='Predicted', ylabel='Actual')
        
        # Add number labels to confusion matrix
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[0,0].text(j, i, format(cm[i, j], 'd'),
                               ha="center", va="center",
                               color="white" if cm[i, j] > thresh else "black")
        
        # 2. Feature importance for classification model
        feature_names = ['total_km', 'avg_km_per_trip', 'trips_per_day', 'vehicle_age_months']
        importances = clf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        axes[0,1].bar(range(len(importances)), importances[indices])
        axes[0,1].set_title('Feature Importance - Classification Model')
        axes[0,1].set_xticks(range(len(importances)))
        axes[0,1].set_xticklabels([feature_names[i] for i in indices], rotation=45)
        
        # 3. Actual vs Predicted for regression model
        axes[1,0].scatter(reg_y_test, reg_y_pred, alpha=0.5)
        axes[1,0].plot([reg_y_test.min(), reg_y_test.max()], [reg_y_test.min(), reg_y_test.max()], 'r--', lw=2)
        axes[1,0].set_xlabel('Actual Remaining KM')
        axes[1,0].set_ylabel('Predicted Remaining KM')
        axes[1,0].set_title('Actual vs Predicted - Regression Model')
        
        # 4. Residuals plot for regression model
        residuals = reg_y_test - reg_y_pred
        axes[1,1].scatter(reg_y_pred, residuals, alpha=0.5)
        axes[1,1].axhline(y=0, color='r', linestyle='--')
        axes[1,1].set_xlabel('Predicted Remaining KM')
        axes[1,1].set_ylabel('Residuals')
        axes[1,1].set_title('Residuals Plot - Regression Model')
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        print("\nModel evaluation chart saved to 'model_evaluation.png'")
        plt.show()
    except ImportError:
        print("\nMatplotlib not installed, skipping visualization.")

def test_models():
    """
    Test models with real-world scenarios
    """
    print("\n--- TESTING MODELS WITH REAL-WORLD SCENARIOS ---")
    
    # Load classification model
    clf_model = joblib.load("models/classification_model.pkl")
    
    # Load regression model
    reg_model = joblib.load("models/regression_model.pkl")
    
    # Test scenarios - representing real-world situations
    test_scenarios = [
        {
            "name": "New bike, low usage",
            "data": [[5000, 20, 2, 6]],  # total_km, avg_km_per_trip, trips_per_day, vehicle_age_months
            "description": "New bike with 5000km, 20km/trip avg, 2 trips/day, 6 months old"
        },
        {
            "name": "Average usage bike",
            "data": [[15000, 25, 4, 12]],
            "description": "Bike with 15000km, 25km/trip avg, 4 trips/day, 12 months old"
        },
        {
            "name": "High usage bike",
            "data": [[30000, 15, 8, 18]],
            "description": "Bike with 30000km, 15km/trip avg, 8 trips/day, 18 months old"
        },
        {
            "name": "Old bike, high usage",
            "data": [[45000, 10, 10, 36]],
            "description": "Bike with 45000km, 10km/trip avg, 10 trips/day, 36 months old"
        },
        {
            "name": "Very old bike, continuous usage",
            "data": [[55000, 8, 15, 48]],
            "description": "Bike with 55000km, 8km/trip avg, 15 trips/day, 48 months old"
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  Description: {scenario['description']}")
        
        # Predict with classification model
        clf_pred = clf_model.predict(scenario['data'])[0]
        clf_prob = clf_model.predict_proba(scenario['data'])[0]
        status_clf = "NEEDS MAINTENANCE NOW" if clf_pred == 1 else "does not need maintenance"
        print(f"  Classification: {status_clf} (Maintenance probability: {clf_prob[1]:.3f})")
        
        # Predict with regression model
        reg_pred = reg_model.predict(scenario['data'])[0]
        status_reg = "NEEDS MAINTENANCE NOW" if reg_pred <= 500 else "does not need maintenance"
        print(f"  Regression: {reg_pred:.2f} km remaining ({status_reg})")

def main():
    """
    Main function to execute the entire workflow
    """
    print("STARTING VEHICLE MAINTENANCE PREDICTION MODEL BUILDING PROCESS")
    print("="*60)
    
    # Train and evaluate classification model
    clf_model, clf_X_test, clf_y_test, clf_y_pred, clf_y_pred_proba = train_and_evaluate_classification_model()
    
    # Train and evaluate regression model
    reg_model, reg_X_test, reg_y_test, reg_y_pred = train_and_evaluate_regression_model()
    
    # Visualize results (if possible)
    try:
        visualize_results(clf_model, clf_X_test, clf_y_test, clf_y_pred, 
                         reg_model, reg_X_test, reg_y_test, reg_y_pred)
    except:
        print("Cannot create visualization due to missing libraries or errors.")
    
    # Test models with real-world scenarios
    test_models()
    
    print("\n" + "="*60)
    print("MODEL BUILDING PROCESS COMPLETED!")
    print("Models have been saved in the 'models/' directory")
    print("- models/classification_model.pkl (classification model)")
    print("- models/regression_model.pkl (regression model)")

if __name__ == "__main__":
    main()