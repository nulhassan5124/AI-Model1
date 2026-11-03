try:
    import joblib
except ImportError:
    from sklearn.utils import _joblib as joblib

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

def main():
    # 1. Load the dataset
    csv_path = '/content/extracted_data/cars.csv'
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        # Fallback to local directory if run locally
        df = pd.read_csv('cars.csv')

    # 2. Prepare clean dataset
    df_clean = df.dropna(subset=['FuelType']).copy()
    features = ['Price', 'Age', 'KM', 'MetColor', 'Automatic', 'CC', 'Weight']
    X = df_clean[features]
    y = df_clean['FuelType']

    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Apply SMOTE to handle class imbalance
    print("Applying SMOTE to training data...")
    smote = SMOTE(k_neighbors=2, random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 5. Train Decision Tree
    print("Training Decision Tree Classifier...")
    clf_smote = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf_smote.fit(X_train_res, y_train_res)

    # 6. Evaluate the model
    y_pred_smote = clf_smote.predict(X_test)
    print(f"\nSMOTE Decision Tree Model Accuracy: {accuracy_score(y_test, y_pred_smote):.2%}\n")
    print("--- Classification Report (After SMOTE) ---")
    print(classification_report(y_test, y_pred_smote))

    # 7. Save the model
    model_filename = 'decision_tree_smote_model.joblib'
    joblib.dump(clf_smote, model_filename)
    print(f"\nModel successfully saved as: {model_filename}")

if __name__ == '__main__':
    main()
