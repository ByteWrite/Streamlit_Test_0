# model_training.py
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def load_data():
    diabetes = load_diabetes()
    df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target
    return df

def train_model(df):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def save_model(model, filepath):
    joblib.dump(model, filepath)

def main():
    print('Loading data...')
    df = load_data()

    print('Training model...')
    model, X_test, y_test = train_model(df)

    print('Model trained!')

    print('Evaluating model...')
    accuracy = evaluate_model(model, X_test, y_test)

    print(f'Model accuracy: {accuracy:.2f}')

    print('Saving model...')
    save_model(model, 'diabetes_prediction_model.pkl')

if __name__ == '__main__':
    main()
