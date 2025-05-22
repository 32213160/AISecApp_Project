# model.py

from sklearn.ensemble import IsolationForest
import joblib

def train_model(X):
    """
    이상 탐지 모델 학습 함수
    
    Args:
        X (DataFrame): 학습 데이터
        
    Returns:
        model: 학습된 모델
    """
    # Isolation Forest 모델 (이상치 탐지에 효과적)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X)
    
    # 모델 저장
    joblib.dump(model, "model.pkl")
    
    print("이상 탐지 모델 학습 완료")
    return model

def predict(model, X):
    """
    이상 탐지 예측 함수
    
    Args:
        model: 학습된 모델
        X (DataFrame): 예측할 데이터
        
    Returns:
        array: 예측 결과 (1: 정상, -1: 이상)
    """
    return model.predict(X)  # 1: 정상, -1: 이상
