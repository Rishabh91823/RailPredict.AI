import pandas as pd
import time
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_model():
    print("Loading 1 Crore dataset from 'railway_dataset_1cr.csv'...")
    start_time = time.time()
    
    df = pd.read_csv('railway_dataset_1cr.csv')
    print(f"Loaded {len(df):,} records in {time.time() - start_time:.2f} seconds.")

    features = ['route_tier', 'days_left', 'waitlist_num', 'train_type', 'class', 'quota', 'is_festival']
    X = df[features]
    y = df['confirmed']

    print("Splitting dataset into training and validation sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training HistGradientBoostingClassifier on 8,000,000 rows...")
    model_start = time.time()
    model = HistGradientBoostingClassifier(
        max_iter=100, 
        learning_rate=0.1, 
        max_depth=12, 
        random_state=42
    )
    model.fit(X_train, y_train)
    print(f"Model training completed in {time.time() - model_start:.2f} seconds.")

    # Model Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy on Test Set: {acc * 100:.2f}%\n")

    # Export model artifact
    output_filename = 'railway_model_1cr.pkl'
    print(f"Exporting trained model to '{output_filename}'...")
    joblib.dump(model, output_filename)
    print(f"Successfully saved '{output_filename}'!")

if __name__ == '__main__':
    train_model()