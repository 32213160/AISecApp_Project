# incremental.py

import pandas as pd
from preprocess import load_data, extract_features
from model import train_model

def retrain_model():
    """
    새로운 데이터로 모델을 재학습하는 함수
    """
    # 데이터 로드 및 전처리
    df = load_data()
    X = extract_features(df)
    
    # 모델 재학습
    model = train_model(X)
    
    print("✅ 모델이 새 데이터로 재학습되었습니다.")
    return model

if __name__ == "__main__":
    retrain_model()
