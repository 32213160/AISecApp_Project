# evaluate.py

from sklearn.metrics import classification_report
from preprocess import load_data, extract_features
from model import train_model, predict

def evaluate():
    """
    모델 성능 평가 함수
    """
    # 데이터 로드 및 전처리
    df = load_data()
    X = extract_features(df)
    
    # 이상으로 간주할 기준 (임계값 기반 라벨링)
    y_true = (X["cpu_temp"] > 80) | (X["fan_speed"] < 500)
    y_true = y_true.astype(int).replace({1: -1, 0: 1})  # 이상: -1, 정상: 1
    
    # 모델 학습 및 예측
    model = train_model(X)
    y_pred = predict(model, X)
    
    # 성능 평가
    print("📊 성능 평가 결과:")
    print(classification_report(y_true, y_pred, target_names=["이상", "정상"]))
    
    return classification_report(y_true, y_pred, target_names=["이상", "정상"], output_dict=True)

if __name__ == "__main__":
    evaluate()
