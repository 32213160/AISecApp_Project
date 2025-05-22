import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from data_storage import save_training_data, load_training_data

def train_model(X_train):
    model = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=42,
        verbose=1
    )
    model.fit(X_train)
    return model

def evaluate_model(model, X_test, y_true=None):
    preds = model.predict(X_test)
    preds = [1 if p == -1 else 0 for p in preds]
    if y_true is not None:
        print(classification_report(y_true, preds))
    return preds

def save_model(model, scaler, model_path='model.pkl', scaler_path='scaler.pkl'):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

def load_model(model_path='model.pkl', scaler_path='scaler.pkl'):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def retrain_model(X_new):
    """재학습 시 기존 데이터 + 신규 데이터 불러와서 학습"""
    X_old = load_training_data()
    if X_old is not None:
        X_combined = np.vstack((X_old, X_new))
    else:
        X_combined = X_new

    model = train_model(X_combined)
    save_training_data(X_combined)
    return model
